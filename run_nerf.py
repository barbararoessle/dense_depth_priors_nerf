import os
import shutil
import subprocess
import math
import time
import datetime
from argparse import Namespace

import configargparse
from skimage.metrics import structural_similarity
from lpips import LPIPS
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from model import NeRF, get_embedder, get_rays, precompute_quadratic_samples, sample_pdf, img2mse, mse2psnr, to8b, \
    compute_depth_loss, select_coordinates, to16b, resnet18_skip
from data import create_random_subsets, load_scene, convert_depth_completion_scaling_to_m, \
    convert_m_to_depth_completion_scaling, get_pretrained_normalize, resize_sparse_depth
from train_utils import MeanTracker, update_learning_rate
from metric import compute_rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, bb_center, bb_scale, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale
    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def precompute_depth_sampling(depth):
    depth_min = (depth[:, 0] - 3. * depth[:, 1])
    depth_max = depth[:, 0] + 3. * depth[:, 1]
    return torch.stack((depth[:, 0], depth_min, depth_max), -1)

def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=25):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_' + filename)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0
    for img_idx in range(len(poses)):
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, with_5_9=True, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)
            max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            depth_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            video_frame = np.concatenate((video_frame, depth_frame), 1)
            depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
            depth_std = depth_var.clamp(0., 1.).sqrt()
            video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)
            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.jpg'), video_frame)

    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
    subprocess.call(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(video_dir, "%d.jpg"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), video_file])
    print("Maximal depth in video: {}".format(max_depth_in_video))

def optimize_camera_embedding(image, pose, H, W, intrinsic, args, render_kwargs_test):
    render_kwargs_test["embedded_cam"] = torch.zeros(args.input_ch_cam, requires_grad=True).to(device)
    optimizer = torch.optim.Adam(params=(render_kwargs_test["embedded_cam"],), lr=5e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    half_W = W
    print(" - Optimize camera embedding")
    max_psnr = 0
    best_embedded_cam = torch.zeros(args.input_ch_cam).to(device)
    # make batches
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, half_W - 1, half_W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2]).long()
    assert(coords[:, 1].max() < half_W)
    batches = create_random_subsets(range(len(coords)), 2 * args.N_rand, device=device)
    # make rays
    rays_o, rays_d = get_rays(H, half_W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    start_time = time.time()
    for i in range(100):
        sum_img_loss = torch.zeros(1)
        optimizer.zero_grad()
        for b in batches:
            curr_coords = coords[b]
            curr_rays_o = rays_o[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            curr_rays_d = rays_d[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            target_s = image[curr_coords[:, 0], curr_coords[:, 1]]
            batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)
            rgb, _, _, _ = render(H, half_W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, **render_kwargs_test)
            img_loss = img2mse(rgb, target_s)
            img_loss.backward()
            sum_img_loss += img_loss
        optimizer.step()
        psnr = mse2psnr(sum_img_loss / len(batches))
        lr_scheduler.step(psnr)
        if psnr > max_psnr:
            max_psnr = psnr
            best_embedded_cam = render_kwargs_test["embedded_cam"].detach().clone()
            print("Step {}: PSNR: {} ({:.2f}min)".format(i, psnr, (time.time() - start_time) / 60))
    render_kwargs_test["embedded_cam"] = best_embedded_cam

def render_images_with_metrics(count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]
        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        if args.input_ch_cam > 0:
            if embedcam_fn is None:
                # use zero embedding at test time or optimize for the latent code
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
                if with_test_time_optimization:
                    optimize_camera_embedding(target, pose, H, W, intrinsic, args, render_kwargs_test)
                    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_latent_codes_" + args.scene_id)
                    os.makedirs(result_dir, exist_ok=True)
                    np.savetxt(os.path.join(result_dir, str(img_idx) + ".txt"), render_kwargs_test["embedded_cam"].cpu().numpy())
            else:
                render_kwargs_test["embedded_cam"] = embedcam_fn(torch.tensor(img_idx, device=device))
        
        with torch.no_grad():
            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
            
            # compute depth rmse
            depth_rmse = compute_rmse(extras['depth_map'][target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)
            
            # compute color metrics
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (extras['depth_map'] / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()
            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                psnr0 = mse2psnr(img_loss0)
                depths0_res[n] = (extras['depth0'] / far).unsqueeze(0).cpu()
                rgbs0_res[n] = torch.clamp(extras['rgb0'], 0, 1).permute(2, 0, 1).cpu()
                metrics.update({"img_loss0" : img_loss0.item(), "psnr0" : psnr0.item()})
            mean_metrics.add(metrics)
    
    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res}
    if 'rgb0' in extras:
        res.update({"rgbs0" : rgbs0_res, "depths0" : depths0_res,})
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False):
    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

def load_checkpoint(args):
    path = os.path.join(args.ckpt_dir, args.expname)
    ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
    return ckpt

def create_nerf(args, scene_render_params):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)
    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)
        model_fine = nn.DataParallel(model_fine).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, embedded_cam, network_fn : run_network(inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    embedded_cam = torch.tensor((), device=device)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'embedded_cam' : embedded_cam,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights

def raw2depth(raw, z_vals, rays_d):
    weights = compute_weights(raw, z_vals, rays_d)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    weights = compute_weights(raw, z_vals, rays_d, noise)
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far):
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)

def perturb_z_vals(z_vals, pytest):
    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand_like(z_vals)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def compute_samples_around_depth(raw, z_vals, rays_d, N_samples, perturb, lower_bound, near, far):
    sampling_depth, sampling_std = raw2depth(raw, z_vals, rays_d)
    sampling_std = sampling_std.clamp(min=lower_bound)
    depth_min = sampling_depth - 3. * sampling_std
    depth_max = sampling_depth + 3. * sampling_std
    return sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far)

def forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest):
    pts_2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_2[...,:,None]
    raw_2 = network_query_fn(pts_2, viewdirs, embedded_cam, network_fn)
    z_vals = torch.cat((z_vals, z_vals_2), -1)
    raw = torch.cat((raw, raw_2), 1)
    z_vals, indices = z_vals.sort()
    raw = torch.gather(raw, 1, indices.unsqueeze(-1).expand_as(raw))
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)
    return {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights}

def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)

    # sample and render rays for dense depth priors for nerf
    N_samples_half = N_samples // 2
    if precomputed_z_samples is not None:
        # compute a lower bound for the sampling standard deviation as the maximal distance between samples
        lower_bound = precomputed_z_samples[-1] - precomputed_z_samples[-2]
    # train time: use precomputed samples along the whole ray and additionally sample around the depth
    if depth_range is not None:
        valid_depth = depth_range[:,0] >= near[0, 0]
        invalid_depth = valid_depth.logical_not()
        # do a forward pass for the precomputed first half of samples
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        if perturb > 0.:
            z_vals = perturb_z_vals(z_vals, pytest)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        z_vals_2 = torch.empty_like(z_vals)
        # sample around the predicted depth from the first half of samples, if the input depth is invalid
        z_vals_2[invalid_depth] = compute_samples_around_depth(raw.detach()[invalid_depth], z_vals[invalid_depth], rays_d[invalid_depth], N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])
        # sample with in 3 sigma of the input depth, if it is valid
        z_vals_2[valid_depth] = sample_3sigma(depth_range[valid_depth, 1], depth_range[valid_depth, 2], N_samples_half, perturb == 0., near[0, 0], far[0, 0])
        return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest)
    # test time: use precomputed samples along the whole ray and additionally sample around the predicted depth from the first half of samples
    elif precomputed_z_samples is not None:
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        z_vals_2 = compute_samples_around_depth(raw, z_vals, rays_d, N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])
        return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest)
    
    # sample and render rays for nerf
    elif not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_ray_batch_from_one_image(H, W, i_train, images, depths, valid_depths, poses, intrinsics, args):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    img_i = np.random.choice(i_train)
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]
    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    if args.depth_loss_weight > 0.:
        depth_range = precompute_depth_sampling(target_d)
        batch_rays = torch.stack([rays_o, rays_d, depth_range], 0)  # (3, N_rand, 3)
    else:
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, target_s, target_d, target_vd, img_i

def complete_depth(images, depths, valid_depths, input_h, input_w, model_path, invalidate_large_std_threshold=-1.):
    device = images.device

    # prepare input
    orig_size = (depths.shape[1], depths.shape[2])
    input_size = (input_h, input_w)
    images_tmp = images.permute(0, 3, 1, 2)
    depths_tmp = depths[..., 0]
    images_tmp = torchvision.transforms.functional.resize(images_tmp, input_size, \
        interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    depths_tmp, valid_depths_tmp = resize_sparse_depth(depths_tmp, valid_depths, input_size)
    normalize, _ = get_pretrained_normalize()
    depths_tmp[valid_depths_tmp] = convert_m_to_depth_completion_scaling(depths_tmp[valid_depths_tmp])
    
    # run depth completion
    with torch.no_grad():
        net = resnet18_skip(pretrained=False, map_location=device, input_size=input_size).to(device)
        net.eval()
        ckpt = torch.load(model_path)
        missing_keys, unexpected_keys = net.load_state_dict(ckpt['network_state_dict'], strict=False)
        print("Loading model: \n  missing keys: {}\n  unexpected keys: {}".format(missing_keys, unexpected_keys))
        
        depths_out = torch.empty_like(depths_tmp)
        depths_std_out = torch.empty_like(depths_tmp)
        for i, (rgb, depth) in enumerate(zip(images_tmp, depths_tmp)):
            rgb = normalize['rgb'](rgb)
            input = torch.cat((rgb, depth.unsqueeze(0)), 0).unsqueeze(0)
            pred = net(input)
            depths_out[i] = convert_depth_completion_scaling_to_m(pred[0])
            depths_std_out[i] = convert_depth_completion_scaling_to_m(pred[1])
        depths_out = torch.stack((depths_out, depths_std_out), 1)
        depths_out = torchvision.transforms.functional.resize(depths_out, orig_size, \
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    
    # apply max min filter
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)
    depths_out_0 = depths_out.narrow(1, 0, 1).clamp(min=0)
    depths_out_max = max_pool(depths_out_0) + 0.01
    depths_out_min = -1. * max_pool(-1. * depths_out_0) - 0.01
    depths_out[:, 1, :, :] = torch.maximum(depths_out[:, 1, :, :], (depths_out_max - depths_out_min).squeeze(1))
    
    # mask out depth with very large uncertainty
    depths_out = depths_out.permute(0, 2, 3, 1)
    valid_depths_out = torch.full_like(valid_depths, True)
    if invalidate_large_std_threshold > 0.:
        large_std_mask = depths_out[:, :, :, 1] > invalidate_large_std_threshold
        valid_depths_out[large_std_mask] = False
        depths_out[large_std_mask] = 0.
        print("Masked out {:.1f} percent of completed depth with standard deviation greater {:.2f}".format( \
            100. * (1. - valid_depths_out.sum() / valid_depths_out.numel()), invalidate_large_std_threshold))
    
    return depths_out, valid_depths_out

def complete_and_check_depth(images, depths, valid_depths, i_train, gt_depths_train, gt_valid_depths_train, scene_sample_params, args):
    near, far = scene_sample_params["near"], scene_sample_params["far"]

    # print statistics before completion
    eval_mask = torch.logical_and(gt_valid_depths_train, valid_depths[i_train])
    p_complete = valid_depths[i_train].sum() * 100. / valid_depths[i_train].numel()
    rmse_before = compute_rmse(depths[i_train, :, :, 0][eval_mask], gt_depths_train.squeeze(-1)[eval_mask])
    print("Depth maps are {:.4f} percent complete and have RMSE {:.4f} before completion".format(p_complete, rmse_before))

    # add channel for depth standard deviation and run depth completion
    depths_std = torch.zeros_like(depths)
    depths = torch.cat((depths, depths_std), 3)
    depths[i_train], valid_depths[i_train] = complete_depth(images[i_train], depths[i_train], valid_depths[i_train], \
        args.depth_completion_input_h, args.depth_completion_input_w, args.depth_prior_network_path, \
        invalidate_large_std_threshold=args.invalidate_large_std_threshold)

    # print statistics after completion
    depths[:, :, :, 0][valid_depths] = depths[:, :, :, 0][valid_depths].clamp(min=near, max=far)
    print("Completed depth maps in range {:.4f} - {:.4f}".format(depths[i_train, :, :, 0][valid_depths[i_train]].min(), \
        depths[i_train, :, :, 0][valid_depths[i_train]].max()))
    eval_mask = torch.logical_and(gt_valid_depths_train, valid_depths[i_train])
    print("Depth maps have RMSE {:.4f} after completion".format(compute_rmse(depths[i_train, :, :, 0][eval_mask], \
        gt_depths_train.squeeze(-1)[eval_mask])))
    lower_bound = 0.03
    depths[:, :, :, 1][valid_depths] = depths[:, :, :, 1][valid_depths].clamp(min=lower_bound, max=(far - near))
    print("Depth standard deviations in range {:.4f} - {:.4f}, with mean {:.4f}".format(depths[i_train, :, :, 1][valid_depths[i_train]].min(), \
        depths[i_train, :, :, 1][valid_depths[i_train]].max(), depths[i_train, :, :, 1][valid_depths[i_train]].mean()))

    return depths, valid_depths

def train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    tb = SummaryWriter(log_dir=os.path.join("runs", args.expname))
    near, far = scene_sample_params['near'], scene_sample_params['far']
    H, W = images.shape[1:3]
    i_train, i_val, i_test, i_video = i_split
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    # use ground truth depth for validation and test if available
    if gt_depths is not None:
        depths[i_test] = gt_depths[i_test]
        valid_depths[i_test] = gt_valid_depths[i_test]
        depths[i_val] = gt_depths[i_val]
        valid_depths[i_val] = gt_valid_depths[i_val]
    i_relevant_for_training = np.concatenate((i_train, i_val), 0)
    if len(i_test) == 0:
        print("Error: There is no test set")
        exit()
    if len(i_val) == 0:
        print("Warning: There is no validation set, test set is used instead")
        i_val = i_test
        i_relevant_for_training = np.concatenate((i_relevant_for_training, i_val), 0)
    # keep test data on cpu until needed
    test_images = images[i_test]
    test_depths = depths[i_test]
    test_valid_depths = valid_depths[i_test]
    test_poses = poses[i_test]
    test_intrinsics = intrinsics[i_test]
    i_test = i_test - i_test[0]

    # move training data to gpu
    images = torch.Tensor(images[i_relevant_for_training]).to(device)
    depths = torch.Tensor(depths[i_relevant_for_training]).to(device)
    valid_depths = torch.Tensor(valid_depths[i_relevant_for_training]).bool().to(device)
    poses = torch.Tensor(poses[i_relevant_for_training]).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)

    # complete and check depth
    gt_depths_train = torch.Tensor(gt_depths[i_train]).to(device) # only used to evaluate error of completed depth
    gt_valid_depths_train = torch.Tensor(gt_valid_depths[i_train]).bool().to(device) # only used to evaluate error of completed depth
    depths, valid_depths = complete_and_check_depth(images, depths, valid_depths, i_train, gt_depths_train, gt_valid_depths_train, \
        scene_sample_params, args)
    del gt_depths_train, gt_valid_depths_train

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, nerf_grad_vars, optimizer = create_nerf(args, scene_sample_params)
    
    # create camera embedding function
    embedcam_fn = None
    if args.input_ch_cam > 0:
        embedcam_fn = torch.nn.Embedding(len(i_train), args.input_ch_cam)

    # optimize nerf
    print('Begin')
    N_iters = 500000 + 1
    global_step = start
    start = start + 1
    for i in trange(start, N_iters):

        # update learning rate
        if i > args.start_decay_lrate and i <= args.end_decay_lrate:
            portion = (i - args.start_decay_lrate) / (args.end_decay_lrate - args.start_decay_lrate)
            decay_rate = 0.1
            new_lrate = args.lrate * (decay_rate ** portion)
            update_learning_rate(optimizer, new_lrate)
        
        # make batch
        batch_rays, target_s, target_d, target_vd, img_i = get_ray_batch_from_one_image(H, W, i_train, images, depths, valid_depths, poses, \
            intrinsics, args)
        if args.input_ch_cam > 0:
            render_kwargs_train['embedded_cam'] = embedcam_fn(torch.tensor(img_i, device=device))
        target_d = target_d.squeeze(-1)

        # render
        rgb, _, _, extras = render(H, W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)
        
        # compute loss and optimize
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        loss = img_loss
        if args.depth_loss_weight > 0.:
            depth_loss = compute_depth_loss(extras['depth_map'], extras['z_vals'], extras['weights'], target_d, target_vd)
            loss = loss + args.depth_loss_weight * depth_loss
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)
            loss = loss + img_loss0
        loss.backward()
        nn.utils.clip_grad_value_(nerf_grad_vars, 0.1)
        optimizer.step()
                
        # write logs
        if i%args.i_weights==0:
            path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        
        if i%args.i_print==0:
            tb.add_scalars('mse', {'train': img_loss.item()}, i)
            if args.depth_loss_weight > 0.:
                tb.add_scalars('depth_loss', {'train': depth_loss.item()}, i)
            tb.add_scalars('psnr', {'train': psnr.item()}, i)
            if 'rgb0' in extras:
                tb.add_scalars('mse0', {'train': img_loss0.item()}, i)
                tb.add_scalars('psnr0', {'train': psnr0.item()}, i)
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {img_loss.item()}  PSNR: {psnr.item()}")
            
        if i%args.i_img==0:
            # visualize 2 train images
            _, images_train = render_images_with_metrics(2, i_train, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, embedcam_fn=embedcam_fn)
            tb.add_image('train_image',  torch.cat((
                torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(8, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
            tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
            tb.add_scalar('ssim', mean_metrics_val.get("ssim"), i)
            tb.add_scalar('lpips', mean_metrics_val.get("lpips"), i)
            if mean_metrics_val.has("depth_rmse"):
                tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
            if 'rgbs0' in images_val:
                tb.add_scalars('mse0', {'val': mean_metrics_val.get("img_loss0")}, i)
                tb.add_scalars('psnr0', {'val': mean_metrics_val.get("psnr0")}, i)
            if 'rgbs0' in images_val:
                tb.add_image('val_image',  torch.cat((
                    torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["rgbs0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)
            else:
                tb.add_image('val_image',  torch.cat((
                    torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)

        # test at the last iteration
        if (i + 1) == N_iters:
            torch.cuda.empty_cache()
            images = torch.Tensor(test_images).to(device)
            depths = torch.Tensor(test_depths).to(device)
            valid_depths = torch.Tensor(test_valid_depths).bool().to(device)
            poses = torch.Tensor(test_poses).to(device)
            intrinsics = torch.Tensor(test_intrinsics).to(device)
            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            write_images_with_metrics(images_test, mean_metrics_test, far, args)
            tb.flush()

        global_step += 1

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test", "test_with_opt", "video"')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default=None, 
                        help='specify the experiment, required for "test" and "video", optional for "train"')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--start_decay_lrate", type=int, default=400000, 
                        help='start iteration for learning rate decay')
    parser.add_argument("--end_decay_lrate", type=int, default=500000, 
                        help='end iteration for learning rate decay')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--depth_loss_weight", type=float, default=0.004,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--invalidate_large_std_threshold", type=float, default=1.,
                        help='invalidate completed depth values with standard deviation greater than threshold in m, \
                            thresholds <=0 deactivate invalidation')
    parser.add_argument("--depth_completion_input_h", type=int, default=240, 
                        help='depth completion network input height')
    parser.add_argument("--depth_completion_input_w", type=int, default=320, 
                        help='depth completion network input width')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=256,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=9,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--input_ch_cam", type=int, default=4,
                        help='number of channels for camera index embedding')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--lindisp", action='store_true', default=False,
                        help='sampling linearly in disparity rather than depth')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=20000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help='checkpoint directory')

    # data options
    parser.add_argument("--scene_id", type=str, default="scene0710_00",
                        help='scene identifier')
    parser.add_argument("--depth_prior_network_path", type=str, default="",
                        help='path to the depth prior network checkpoint to be used')
    parser.add_argument("--data_dir", type=str, default="",
                        help='directory containing the scenes')

    return parser

def run_nerf():
    
    parser = config_parser()
    args = parser.parse_args()

    

    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)
    else:
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
        tmp_task = args.task
        tmp_data_dir = args.data_dir
        tmp_ckpt_dir = args.ckpt_dir
        # load nerf parameters from training
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        with open(args_file, 'r') as af:
            args_dict = json.load(af)
        args = Namespace(**args_dict)
        # task and paths are not overwritten
        args.task = tmp_task
        args.data_dir = tmp_data_dir
        args.ckpt_dir = tmp_ckpt_dir

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)
    images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths = load_scene(scene_data_dir)

    i_train, i_val, i_test, i_video = i_split

    # Compute boundaries of 3D space
    max_xyz = torch.full((3,), -1e6)
    min_xyz = torch.full((3,), 1e6)
    for idx_train in i_train:
        rays_o, rays_d = get_rays(H, W, torch.Tensor(intrinsics[idx_train]), torch.Tensor(poses[idx_train])) # (H, W, 3), (H, W, 3)
        points_3D = rays_o + rays_d * far # [H, W, 3]
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
    args.bb_center = (max_xyz + min_xyz) / 2.
    args.bb_scale = 2. / (max_xyz - min_xyz).max()
    print("Computed scene boundaries: min {}, max {}".format(min_xyz, max_xyz))

    # Precompute scene sampling parameters
    if args.depth_loss_weight > 0.:
        precomputed_z_samples = precompute_quadratic_samples(near, far, args.N_samples // 2)

        if precomputed_z_samples.shape[0] % 2 == 1:
            precomputed_z_samples = precomputed_z_samples[:-1]

        print("Computed {} samples between {} and {}".format(precomputed_z_samples.shape[0], precomputed_z_samples[0], precomputed_z_samples[-1]))
    else:
        precomputed_z_samples = None
    scene_sample_params = {
        'precomputed_z_samples' : precomputed_z_samples,
        'near' : near,
        'far' : far,
    }

    lpips_alex = LPIPS()

    if args.task == "train":
        train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths)
        exit()

    # create nerf model for testing
    _, render_kwargs_test, _, nerf_grad_vars, _ = create_nerf(args, scene_sample_params)
    for param in nerf_grad_vars:
        param.requires_grad = False

    # render test set and compute statistics
    if "test" in args.task: 
        with_test_time_optimization = False
        if args.task == "test_opt":
            with_test_time_optimization = True
        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)
        write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization)
    elif args.task == "video":
        vposes = torch.Tensor(poses[i_video]).to(device)
        vintrinsics = torch.Tensor(intrinsics[i_video]).to(device)
        render_video(vposes, H, W, vintrinsics, str(0), args, render_kwargs_test)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run_nerf()
