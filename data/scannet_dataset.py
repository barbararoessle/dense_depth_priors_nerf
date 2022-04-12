import os
import random
import math
import sqlite3

import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms

from .error_sources import add_missing_depth, add_quadratic_depth_noise

def is_in_list(file, list_to_check):
    for h in list_to_check:
        if h in file:
            return True
    return False

def get_whitelist(dataset_dir, dataset_split):
    whitelist_txt = os.path.join(dataset_dir, "scannetv2_{}.txt".format(dataset_split))
    scenes = pd.read_csv(whitelist_txt, names=["scenes"], header=None)
    return scenes["scenes"].tolist()

def apply_filter(files, dataset_dir, dataset_split):
    whitelist = get_whitelist(dataset_dir, dataset_split)
    return [f for f in files if is_in_list(f, whitelist)]

def read_rgb(rgb_file):
    bgr = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert rgb.shape[2] == 3

    to_tensor = transforms.ToTensor()
    rgb = to_tensor(rgb)
    return rgb

def read_depth(depth_file):
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    assert len(depth.shape) == 2

    valid_depth = depth.astype('bool')
    depth = depth.astype('float32')

    # 16bit integer range corresponds to range 0 .. 65.54m
    # use the first quarter of this range up to 16.38m and invalidate depth values beyond
    # scale depth, such that range 0 .. 1 corresponds to range 0 .. 16.38m
    max_depth = np.float32(2 ** 16 - 1) / 4.
    depth = depth / max_depth
    invalidate_mask = depth > 1.
    depth[invalidate_mask] = 0.
    valid_depth[invalidate_mask] = False
    return transforms.functional.to_tensor(depth), transforms.functional.to_tensor(valid_depth)

def convert_depth_completion_scaling_to_m(depth):
    # convert from depth completion scaling to meter, that means map range 0 .. 1 to range 0 .. 16,38m
    return depth * (2 ** 16 - 1) / 4000.

def convert_m_to_depth_completion_scaling(depth):
    # convert from meter to depth completion scaling, which maps range 0 .. 16,38m to range 0 .. 1
    return depth * 4000. / (2 ** 16 - 1)

def get_normalize(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    unnormalize = transforms.Normalize(mean=np.divide(-mean, std), std=(1. / std))
    return normalize, unnormalize

def get_pretrained_normalize():
    normalize = dict()
    unnormalize = dict()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize['rgb'], unnormalize['rgb'] = get_normalize(mean, std)
    normalize['rgbd'], unnormalize['rgbd'] = get_normalize(np.concatenate((mean, [0.,]), axis=0), np.concatenate((std, [1.,]), axis=0))
    return normalize, unnormalize

def resize_sparse_depth(depths, valid_depths, size):
    device = depths.device
    orig_size = (depths.shape[1], depths.shape[2])
    col, row = torch.meshgrid(torch.tensor(range(orig_size[1])), torch.tensor(range(orig_size[0])), indexing='ij')
    rowcol2rowcol = torch.stack((row.t(), col.t()), -1)
    rowcol2rowcol = rowcol2rowcol.unsqueeze(0).expand(depths.shape[0], -1, -1, -1)
    image_index = torch.arange(depths.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, orig_size[0], orig_size[1], 1)
    rowcol2rowcol = torch.cat((image_index, rowcol2rowcol), -1)
    factor_h, factor_w = float(size[0]) / float(orig_size[0]), float(size[1]) / float(orig_size[1])
    depths_out = torch.zeros((depths.shape[0], size[0], size[1]), device=device)
    valid_depths_out = torch.zeros_like(depths_out).bool()
    idx_row_col = rowcol2rowcol[valid_depths]
    idx_row_col_resized = idx_row_col
    idx_row_col_resized = ((idx_row_col + 0.5) * torch.tensor((1., factor_h, factor_w))).long() # consider pixel centers
    depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] \
        = depths[idx_row_col[..., 0], idx_row_col[..., 1], idx_row_col[..., 2]]
    valid_depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] = True
    return depths_out, valid_depths_out

class ScanNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir, data_split, db_path, random_rot=0, load_size=(240, 320), \
            horizontal_flip=False, color_jitter=None, depth_noise=False, missing_depth_percent=0.998):
        super(ScanNetDataset, self).__init__()

        # apply train val test split
        self.dataset_dir = dataset_dir
        dir_suffix = ""
        if data_split == "test":
            dir_suffix = "_test"
        input_scenes_dir = "scans{}".format(dir_suffix)
        filtered_scenes = [os.path.join(input_scenes_dir, s) for s in 
            apply_filter(os.listdir(os.path.join(dataset_dir, input_scenes_dir)), dataset_dir, data_split)]
        
        # create file list
        self.rgb_files = []
        for rel_scene_path in filtered_scenes:
            rel_scene_color_path = os.path.join(rel_scene_path, "color")
            for rgb in os.listdir(os.path.join(dataset_dir, rel_scene_color_path)):
                rel_rgb_path = os.path.join(rel_scene_color_path, rgb)
                self.rgb_files.append(rel_rgb_path)
        
        # transformation
        self.normalize, self.unnormalize = get_pretrained_normalize()
        self.random_rot = random_rot
        self.load_size = load_size
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter

        # depth sampling
        self.missing_depth_percent = missing_depth_percent # add percentage of missing depth
        self.depth_noise = depth_noise # add gaussian depth noise
        # open keypoint database for sampling at image keypoints
        self.feature_db = sqlite3.connect(db_path).cursor()
        self.id2dbid = dict((n[:-4], id) for n, id in self.feature_db.execute("SELECT name, image_id FROM images"))

    def __getitem__(self, index):
        rgb_file = os.path.join(self.dataset_dir, self.rgb_files[index])
        depth_file = rgb_file.replace("color", "depth").replace(".jpg", ".png")
        rgb = read_rgb(rgb_file)
        depth, valid_depth = read_depth(depth_file)
        # pad to make aspect ratio of rgb (968x1296) and depth (480x640) match
        if rgb.shape[1] == 968 and rgb.shape[2] == 1296:
            # pad 2 pixels on both sides in height dimension
            pad_rgb_height = 2
            rgb = torch.nn.functional.pad(rgb, (0, 0, pad_rgb_height, pad_rgb_height))
            depth_shape = depth.shape
            rgb_shape = rgb.shape
            scale_rgb = (float(depth_shape[1]) / float(rgb_shape[1]), float(depth_shape[2]) / float(rgb_shape[2]))
            rgb = transforms.functional.resize(rgb, (depth_shape[1], depth_shape[2]), interpolation=transforms.functional.InterpolationMode.NEAREST)
        else:
            pad_rgb_height = 0.
            scale_rgb = (1., 1.)
        id = self.rgb_files[index][:-4].replace("scans_test/", "").replace("scans/", "")
        
        # precompute random rotation
        rot = random.uniform(-self.random_rot, self.random_rot)

        # precompute resize and crop
        tan_abs_rot = math.tan(math.radians(abs(rot)))
        border_width = math.ceil(self.load_size[0] * tan_abs_rot)
        border_height = math.ceil(self.load_size[1] * tan_abs_rot)
        top = math.floor(0.5 * border_height)
        left = math.floor(0.5 * border_width)
        resize_size = (self.load_size[0] + border_height, self.load_size[1] + border_width)

        # precompute random horizontal flip
        apply_hflip = self.horizontal_flip and random.random() > 0.5

        # create a sparsified depth and a complete target depth
        target_valid_depth = valid_depth.clone()
        target_depth = depth.clone()
        depth, valid_depth = self.sample_depth_at_image_features(depth, valid_depth, id, scale_rgb, pad_rgb_height)
        depth, valid_depth = add_missing_depth(depth, valid_depth, self.missing_depth_percent)
        
        rgbd = torch.cat((rgb, depth), 0)
        data = {'rgbd': rgbd, 'valid_depth' : valid_depth, 'target_depth' : target_depth, 'target_valid_depth' : target_valid_depth}

        # apply transformation
        for key in data.keys():
            # resize
            if key == 'rgbd':
                # resize such that sparse points are preserved
                B_depth, data['valid_depth'] = resize_sparse_depth(data['rgbd'][3, :, :].unsqueeze(0), data['valid_depth'], resize_size)
                B_rgb = transforms.functional.resize(data['rgbd'][:3, :, :], resize_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
                data['rgbd'] = torch.cat((B_rgb, B_depth), 0)
            else:
                # avoid blurring the depth channel with invalid values by using interpolation mode nearest
                data[key] = transforms.functional.resize(data[key], resize_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
            
            # augment color
            if key == 'rgbd':
                if self.color_jitter is not None:
                    cj = transforms.ColorJitter(brightness=self.color_jitter, contrast=self.color_jitter, saturation=self.color_jitter, \
                        hue=self.color_jitter)
                    data['rgbd'][:3, :, :] = cj(data['rgbd'][:3, :, :])
            
            # rotate
            if self.random_rot != 0:
                data[key] = transforms.functional.rotate(data[key], rot)
            
            # crop
            data[key] = transforms.functional.crop(data[key], top, left, self.load_size[0], self.load_size[1])

            # horizontal flip
            if apply_hflip:
                data[key] = transforms.functional.hflip(data[key])
            
            # normalize
            if key == 'rgbd':
                data[key] = self.normalize['rgbd'](data[key])
                # scale depth according to resizing due to rotation
                data[key][3, :, :] /= (1. + tan_abs_rot)

        # add depth noise
        if self.depth_noise:
            data['rgbd'][3, :, :] = convert_m_to_depth_completion_scaling(add_quadratic_depth_noise( \
                convert_depth_completion_scaling_to_m(data['rgbd'][3, :, :]), data['valid_depth'].squeeze()))

        return data

    def sample_depth_at_image_features(self, depth, valid_depth, id, scale, pad_height):
        depth_shape = depth.shape
        db_id = self.id2dbid[id]
        # 6 affine coordinates
        keypoints = [np.frombuffer(coords[0], dtype=np.float32).reshape(-1, 6) if coords[0] is not None else None for coords in self.feature_db.execute( \
            "SELECT data FROM keypoints WHERE image_id=={}".format(db_id))]
        if keypoints[0] is not None:
            cols = keypoints[0][:, 0]
            rows = keypoints[0][:, 1]
            rows = rows + pad_height
            cols = (cols * scale[1]).astype(int)
            rows = (rows * scale[0]).astype(int)
            row_col_mask = (rows >= 0) & (rows < depth_shape[1]) & (cols >= 0) & (cols < depth_shape[2])
            rows = rows[row_col_mask]
            cols = cols[row_col_mask]
            keypoints_mask = torch.full(depth_shape, False)
            keypoints_mask[0, rows, cols] = True
            valid_depth = torch.logical_and(keypoints_mask, valid_depth)
            depth[torch.logical_not(valid_depth)] = 0.
        else:
            depth = torch.zeros_like(depth)
            valid_depth = torch.zeros_like(valid_depth)
        return depth, valid_depth

    def __len__(self):
        return len(self.rgb_files)
