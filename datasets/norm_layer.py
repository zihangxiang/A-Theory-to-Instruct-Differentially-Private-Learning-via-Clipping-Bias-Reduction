import torch
import torch.nn as nn
import torch.nn.functional as F
# from opacus.grad_sample import register_grad_sampler
import torch.nn.init as init

class smart_batchnorm(nn.BatchNorm2d):
        
    def __init__(self, 
                 num_features, 
                 eps = 1e-5, 
                 momentum = 0.1,
                 affine = False, 
                 track_running_stats = False):
        super(smart_batchnorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.gn = nn.GroupNorm(min(4,num_features), num_features, affine =  False)

    def forward(self, x):
        
        return self.gn(x)


    
class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super().__init__(in_channels, out_channels, kernel_size, **kwargs)
  def forward(self, x):        
    weight = self.weight
    weight_mean = weight.mean(dim=(1,2,3), keepdim=True)
    std = weight.std(dim=(1,2,3), keepdim=True) + 1e-6
    weight = (weight - weight_mean)/ std / (weight.numel() / weight.size(0))**0.5
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
