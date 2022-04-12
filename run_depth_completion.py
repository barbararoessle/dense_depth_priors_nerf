import time
import datetime
import os.path
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from train_utils import MeanTracker
import cv2

from data import ScanNetDataset, convert_depth_completion_scaling_to_m, create_random_subsets
from train_utils import print_network_info, get_hours_mins, MeanTracker, make_image_grid, apply_max_filter, \
    update_learning_rate
from model import resnet18_skip
from metric import compute_rmse

def write_batch(batch, path):
    bgr = cv2.cvtColor((batch.permute(1, 2, 0).numpy() * 255.).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr_width = bgr.shape[1] // 5
    depth_columns = cv2.applyColorMap(bgr[:, bgr_width:, :], cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path, np.concatenate((bgr[:, :bgr_width, :], depth_columns), 1))

def make_grid(input, pred_x, pred_std, target, unnormalize):
    input_grid = make_image_grid(input, unnormalize)
    pred_x_grid = make_image_grid(pred_x)
    pred_std_grid = make_image_grid(pred_std)
    target_grid = make_image_grid(target)
    return torch.cat((input_grid, pred_x_grid, pred_std_grid, target_grid), 2)

def batch2grid(input, pred, target, unnormalize, n_samples):
    input = input[:n_samples, ...]
    pred_x = pred[0][:n_samples, ...]
    target = target[:n_samples, ...]
    # clamp at 0.5m and normalize
    pred_std = convert_depth_completion_scaling_to_m(pred[1][:n_samples, ...]).clamp(max=0.5) / 0.5
    return make_grid(apply_max_filter(input, 3), pred_x, pred_std, target, unnormalize)

def get_load_path(args):
    return os.path.join(args.exp_dir, args.expname + '.tar')

def load_net(args):
    load_path = get_load_path(args)
    if os.path.exists(load_path):
        load_pretrained = False
    else:
        load_pretrained = True

    net = resnet18_skip(pretrained=load_pretrained, pretrained_path=args.pretrained_resnet_path)

    print_network_info(net)

    if not load_pretrained:
        ckpt = torch.load(load_path)
        missing_keys, unexpected_keys = net.load_state_dict(ckpt['network_state_dict'], strict=False)
        print("Loading model: \n  missing keys: {}\n  unexpected keys: {}".format(missing_keys, unexpected_keys))

    return net

def load_train_state(args, optimizer):
    load_path = get_load_path(args)
    start_epoch = 1
    min_val_rmse = 1e6
    if os.path.exists(load_path):
        ckpt = torch.load(load_path)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr' in ckpt:
            new_lr = ckpt['lr']
            update_learning_rate(optimizer, new_lr)
            print("Set learning rate to {}".format(new_lr))
        
        start_epoch = ckpt['epoch'] + 1
    return optimizer, start_epoch, min_val_rmse

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    return device

class Validator:
    def __init__(self, val_dataset, unnormalize, min_val_rmse, device):
        self.device = device
        self.unnormalize = unnormalize
        self.min_val_rmse = min_val_rmse
        validate_on_at_least_n_samples = 20000
        val_sample_count = len(val_dataset)
        if val_sample_count < validate_on_at_least_n_samples:
            self.val_subsets = [val_dataset,]
            print("Small validation set -> no need to create subsets")
        else:
            self.val_subsets = create_random_subsets(val_dataset, validate_on_at_least_n_samples)
            print("Create {} validation subsets with length {} or {}".format(len(self.val_subsets), len(self.val_subsets[0]), \
                len(self.val_subsets[-1])))
        self.val_subset_index = 0
    
    def next_subset_index(self):
        curr_subset_index = self.val_subset_index
        self.val_subset_index += 1
        if self.val_subset_index == len(self.val_subsets):
            self.val_subset_index = 0
        return curr_subset_index

    def validate(self, net, optimizer, args, tb, epoch, step):
        with torch.no_grad():
            net.eval()
            val_metrics = MeanTracker()
            val_start_time = time.time()
            for i, data in enumerate(DataLoader(dataset=self.val_subsets[self.next_subset_index()], batch_size=args.batch_size, \
                    shuffle=False, num_workers=4, drop_last=True)):
                batch_start_time = time.time()

                # move data to gpu and predict
                valid_target = data['target_valid_depth'].to(self.device)
                if valid_target.sum() <= 0:
                    continue
                input = data['rgbd'].to(self.device)
                target = data['target_depth'].to(self.device)
                pred = net(input)

                # compute metrics
                val_l1_loss = convert_depth_completion_scaling_to_m(torch.nn.functional.l1_loss(pred[0][valid_target], target[valid_target]))
                val_rmse = convert_depth_completion_scaling_to_m(compute_rmse(pred[0][valid_target], target[valid_target]))
                val_loss = 0.01 * torch.nn.functional.gaussian_nll_loss(pred[0][valid_target], target[valid_target], pred[1][valid_target].pow(2))
                curr_val_metrics = {"l1" : val_l1_loss.item(), "rmse" : val_rmse.item(), "gnll" : val_loss.item(), \
                    "batch_time" : time.time() - batch_start_time}
                val_metrics.add(curr_val_metrics)

                # visualize the first batch
                if i == 0:
                    batch_grid = batch2grid(input, pred, target, self.unnormalize, 8)
                    tb.add_image('val_image',  batch_grid, step)

            # print statistics
            mean_it_time = (time.time() - val_start_time) / (i + 1)
            mean_val_rmse = val_metrics.get("rmse")
            mean_val_l1_loss = val_loss = val_metrics.get("l1")
            tb.add_scalars('l1', {'val': mean_val_l1_loss}, step)
            mean_val_gnll = val_loss = val_metrics.get("gnll")
            tb.add_scalars('gnll', {'val': mean_val_gnll}, step)
            tb.add_scalar('rmse', mean_val_rmse, step)
            print("Validate, it_time={:.3f}s, batch_time={:.3f}s, val_metric={:.4f}".format(mean_it_time, val_metrics.get("batch_time"), val_loss))

            # save checkpoint
            if mean_val_rmse < self.min_val_rmse:
                self.min_val_rmse = mean_val_rmse
                filename = args.expname + '.tar'
                os.makedirs(os.path.join(args.exp_dir), exist_ok=True)
                path = os.path.join(args.exp_dir, filename)
                save_dict = {
                    'epoch': epoch,
                    'lr' : optimizer.param_groups[0]['lr'],
                    'mean_val_rmse': mean_val_rmse,
                    'network_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}
                torch.save(save_dict, path)
                print('Saved checkpoints at', path)
        net.train()
        return val_loss

def train_depth_completion(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = get_device()

    # load network and optimizer
    net = load_net(args).to(device)
    
    optimizer = torch.optim.Adam(list(net.parameters()), lr=args.lr)
    optimizer, start_epoch, min_val_rmse = load_train_state(args, optimizer)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    
    tb = SummaryWriter(log_dir=os.path.join("runs", args.expname))

    # create datasets
    train_dataset = ScanNetDataset(args.dataset_dir, "train", args.db_path, random_rot=args.random_rot, horizontal_flip=True, \
        color_jitter=args.color_jitter, depth_noise=True, missing_depth_percent=args.missing_depth_percent)
    val_dataset = ScanNetDataset(args.dataset_dir, "val", args.db_path, depth_noise=True, missing_depth_percent=args.missing_depth_percent)
    unnormalize = train_dataset.unnormalize
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)
    args.i_val = min(args.i_val, len(train_loader))
    print("Train on {} samples".format(len(train_dataset)))
    validator = Validator(val_dataset, unnormalize, min_val_rmse, device)
    print("Validate on {} samples".format(len(val_dataset)))

    # start training
    train_batch_count = len(train_loader)
    train_metrics = MeanTracker()
    for epoch in range(start_epoch, args.n_epochs + 1): 
        net.train() # switch to train mode
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            step = (epoch - 1) * train_batch_count + i + 1

            # move data to gpu and predict
            valid_target = data['target_valid_depth'].to(device)
            if valid_target.sum() <= 0:
                continue
            input = data['rgbd'].to(device)
            target = data['target_depth'].to(device)
            pred = net(input)

            # compute loss and metrics, update network parameters
            l1_loss = torch.nn.functional.l1_loss(pred[0][valid_target], target[valid_target])
            curr_train_metrics = {"l1" : convert_depth_completion_scaling_to_m(l1_loss.item()),}
            train_loss = 0.01 * torch.nn.functional.gaussian_nll_loss(pred[0][valid_target], target[valid_target], pred[1][valid_target].pow(2))
            curr_train_metrics["gnll"] = train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            curr_train_metrics["batch_time"] = time.time() - batch_start_time
            train_metrics.add(curr_train_metrics)

            # log results
            if (i+1)%args.i_print == 0:
                mean_it_time = (time.time() - epoch_start_time) / (i + 1)
                portion_of_epoch = (i + 1) / float(train_batch_count)
                hours, mins = get_hours_mins(epoch_start_time, time.time())
                print("Epoch {}/{}: {:.2f}% in {:02d}:{:02d}, it_time={:.3f}s, batch_time={:.3f}s, l1={:.4f}".format(epoch, args.n_epochs, \
                    100. * portion_of_epoch, hours, mins, mean_it_time, train_metrics.get("batch_time"), train_metrics.get("l1")))
                tb.add_scalars('l1', {'train': train_metrics.get("l1")}, step)
                tb.add_scalars('gnll', {'train': train_metrics.get("gnll")}, step)
                train_metrics.reset()

            if (i+1)%args.i_img == 0:
                batch_grid = batch2grid(input, pred, target, unnormalize, 8)
                tb.add_image('train_image',  batch_grid, step)
            
            if (i+1)%args.i_val == 0:
                val_loss = validator.validate(net, optimizer, args, tb, epoch, step)
                # update lr
                scheduler.step(val_loss)

    tb.flush()

def main():
    parser = ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test"')
    parser.add_argument("--expname", type=str, default=None, \
        help='specify the experiment, required for "test" or to resume "train"')

    # data
    parser.add_argument("--dataset_dir", type=str, default="", \
        help="dataset directory")
    parser.add_argument("--db_path", type=str, default="scannet_sift_database.db", \
        help='path to the sift database')
    parser.add_argument("--pretrained_resnet_path", type=str, default="resnet18.pth", \
        help='path to the pretrained resnet weights')
    parser.add_argument("--ckpt_dir", type=str, default="", \
        help='checkpoint directory')
    
    # training
    parser.add_argument("--missing_depth_percent", type=float, default=0.998, \
        help='portion of missing depth in sparse depth input, value between 0 and 1')
    parser.add_argument("--random_rot", type=float, default=10., \
        help='random rotation in degree as data augmentation')
    parser.add_argument("--color_jitter", type=float, default=0.4, \
        help='add color jitter as data augmentation, set None to deactivate')
    parser.add_argument("--batch_size", type=int, default=8, \
        help='batch size')
    parser.add_argument("--n_epochs", type=int, default=12, \
        help='number of epochs')
    parser.add_argument("--lr", type=float, default=1e-4, \
        help='learning rate')

    # logging
    parser.add_argument("--i_print",   type=int, default=1000, \
                        help='log train loss every ith batch')
    parser.add_argument("--i_img",     type=int, default=10000, \
                        help='log train images every ith batch')
    parser.add_argument("--i_val", type=int, default=25000, \
                        help='validate every ith batch or every epoch if the train set is smaller')
    
    args = parser.parse_args()

    print(args)

    if args.expname is None:
        args.expname = "{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))
    args.exp_dir = os.path.join(args.ckpt_dir, args.expname)

    device = get_device()

    if args.task == "test":
        # load network weights
        net = load_net(args).to(device)

        result_dir = os.path.join(args.exp_dir, "test_results")
        os.makedirs(os.path.join(result_dir), exist_ok=True)
        
        # create dataset
        test_dataset = ScanNetDataset(args.dataset_dir, "test", args.db_path, depth_noise=True, missing_depth_percent=args.missing_depth_percent)
        unnormalize = test_dataset.unnormalize
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
        print("Test on {} samples".format(len(test_dataset)))
        
        visu_sample_count = len(test_dataset)
        number_visu_images = 40 # number of images to visualize
        visu_samples = range(0, visu_sample_count, visu_sample_count // number_visu_images)
        visu_loader = DataLoader(dataset=Subset(test_dataset, visu_samples), batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
        
        with torch.no_grad():
            net.eval()
            test_metrics = MeanTracker()
            for i, data in enumerate(test_loader):
                
                # move data to gpu and predict
                valid_target = data['target_valid_depth'].to(device)
                if valid_target.sum() <= 0:
                    continue
                input = data['rgbd'].to(device)
                target = data['target_depth'].to(device)
                pred = net(input)

                # compute test metrics
                pred_depth_m = convert_depth_completion_scaling_to_m(pred[0])
                valid_pred_depth_m = pred_depth_m[valid_target]
                target_depth_m = convert_depth_completion_scaling_to_m(target[valid_target])
                mae = torch.nn.functional.l1_loss(valid_pred_depth_m, target_depth_m)
                rmse = compute_rmse(valid_pred_depth_m, target_depth_m)
                curr_metrics = {"mae" : mae.item(), "rmse" : rmse.item()}
                pred_std_m = convert_depth_completion_scaling_to_m(pred[1])
                curr_metrics["std"] = pred_std_m.mean()
                test_metrics.add(curr_metrics)
                if (i % 1000) == 0:
                    print("{}/{}".format(i, len(test_loader)))
        
            with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
                test_metrics.print(f)
            test_metrics.print()

            # write visualization samples
            for i, data in enumerate(visu_loader):
                valid_target = data['target_valid_depth'].to(device)
                input = data['rgbd'].to(device)
                target = data['target_depth'].to(device)
                pred = net(input)
                batch_grid = batch2grid(input, pred, target, unnormalize, args.batch_size)
                write_batch(batch_grid.cpu(), os.path.join(result_dir, str(i) + ".jpg"))
        exit()
    else:
        train_depth_completion(args)

if __name__ == "__main__":
    main()