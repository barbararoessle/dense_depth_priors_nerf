from scipy import ndimage
import torch
import torchvision

class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (self.mean_dict[key] * self.total_weight + l) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return (key in self.mean_dict)

    def get(self, key):
        return self.mean_dict[key]
    
    def as_dict(self):
        return self.mean_dict
        
    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0
    
    def print(self, f=None):
        for key, l in self.mean_dict.items():
            if f is not None:
                print("{}: {}".format(key, l), file=f)
            else:
                print("{}: {}".format(key, l))

def get_hours_mins(start_time, end_time):
    dt = end_time - start_time
    hours = int(dt // 3600)
    mins = int((dt // 60) % 60)
    return hours, mins

def apply_max_filter(batch, channel, kernel=3):
    batch_local = batch.detach().clone()
    for i, image in enumerate(batch_local.cpu().numpy()):
        batch_local[i, channel, :, :] = torch.tensor(ndimage.maximum_filter(image[channel, :, :], \
            size=kernel)).type(torch.FloatTensor)
    return batch_local

def make_image_grid(data, unnormalize=None):
    if data.shape[1] == 1:
        return torchvision.utils.make_grid(data, nrow=1)
    elif data.shape[1] == 3:
        return torchvision.utils.make_grid(data if unnormalize is None else unnormalize['rgb'](data), nrow=1)
    elif data.shape[1] == 4:
        unnormalized = data if unnormalize is None else unnormalize['rgbd'](data)
        rgb_grid = torchvision.utils.make_grid(unnormalized[:, :3, :, :], nrow=1)
        depth_grid = torchvision.utils.make_grid(torch.unsqueeze(unnormalized[:, 3, :, :], 1), nrow=1)
        return torch.cat((rgb_grid, depth_grid), 2)

def print_network_info(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Number of model parameters: %.3f M' % (num_params / 1e6))
