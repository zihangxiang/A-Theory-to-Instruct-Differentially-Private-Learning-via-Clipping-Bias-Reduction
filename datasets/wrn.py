import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

#
from . import norm_layer as nl

class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super().__init__(in_channels, out_channels, kernel_size, **kwargs)
  def forward(self, x):        
    weight = self.weight
    weight_mean = weight.mean(dim=(1,2,3), keepdim=True)
    std = weight.std(dim=(1,2,3), keepdim=True) + 1e-5
    weight = (weight - weight_mean)/ std
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



norm_layer = nl.smart_batchnorm
# smart_batchnorm = nn.BatchNorm2d

# conv_layer = partial(nn.Conv2d, bias = False)
conv_layer = partial(Conv2d, bias = False)
NL = nn.ELU

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.relu1 = NL(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = NL(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and conv_layer(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
        self.relu3 = NL(inplace=True)
        self.bn3 = norm_layer(out_planes)

    def forward(self, x):
        ''' none_linear vs norm layer order'''

        
        if not self.equalInOut:
            x = self.bn1(self.relu1(x))
        else:
            out = self.bn1(self.relu1(x))
        out = self.bn2( self.relu2(self.conv1(out if self.equalInOut else x)) )
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), self.conv2(out))
        return self.bn3(self.relu3(out))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        base_channel = 8
        nChannels = [base_channel, base_channel*widen_factor, 2*base_channel*widen_factor, 4*base_channel*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = conv_layer(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = norm_layer(nChannels[3])
        self.relu = NL(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, norm_layer):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        # print(f'x shape: {x.shape}')
        out = F.elu( self.conv1(x) )
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        ''' none_linear vs norm layer order'''
        # out = self.relu(self.bn1(out))
        out = self.bn1(self.relu(out))
        
        
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
