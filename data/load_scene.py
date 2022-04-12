import os

import numpy as np
import json
import cv2

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

def load_ground_truth_depth(basedir, train_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in train_filenames:
        filename = filename.replace("rgb", "target_depth")
        filename = filename.replace(".jpg", ".png")
        gt_depth_fname = os.path.join(basedir, filename)
        if os.path.exists(gt_depth_fname):
            gt_depth = cv2.imread(gt_depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_valid_depth = gt_depth > 0.5
            gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)
        else:
            gt_depth = np.zeros((H, W))
            gt_valid_depth = np.full_like(gt_depth, False)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths

def load_scene(basedir):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = float(meta['near'])
                far = float(meta['far'])
                depth_scaling_factor = float(meta['depth_scaling_factor'])
           
            imgs = []
            depths = []
            valid_depths = []
            poses = []
            intrinsics = []
            
            for frame in meta['frames']:
                if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
                    img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
                    
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)

                    filenames.append(frame['file_path'])
                    
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)

                poses.append(np.array(frame['transform_matrix']))
                H, W = img.shape[:2]
                fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
                all_depths.append(np.array(depths))
                all_valid_depths.append(np.array(valid_depths))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])
        
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    gt_depths, gt_valid_depths = load_ground_truth_depth(basedir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths
