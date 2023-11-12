import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from . import norm_layer as nl


smart_batchnorm = nl.smart_batchnorm

conv_layer = partial(nl.Conv2d, bias = False)


''' 3 places are modified: base block, resblock, conv, final linear layer '''

''' the following is the resnet implementation '''
def _weights_init(m):
    # classname = m.__class__.__name__
    # #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, )
        self.bn1 = smart_batchnorm(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = smart_batchnorm(planes)
        self.bn3 = smart_batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                     smart_batchnorm(self.expansion * planes)
                )


    def forward(self, x):
       
        
        ''' norm layer after activation '''
        NLA = F.elu
        out = self.bn1(NLA(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.bn3( NLA(out) )
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = smart_batchnorm(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
       

        out = self.bn1( F.elu( self.conv1(x) ) )
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        ''''''
        out = (out - out.mean(dim=1, keepdim=True)) / (out.std(dim=1, keepdim=True)+ 1e-6)
        out = self.linear(out)
        
        return out



''' model candidates '''
def resnet14(num_class):
    return ResNet(BasicBlock, [2, 2, 2], num_class)

def resnet20(num_class):
    return ResNet(BasicBlock, [3, 3, 3], num_class)

def resnet22(num_class):
    return ResNet(BasicBlock, [3, 4, 3], num_class)

def resnet26(num_class):
    return ResNet(BasicBlock, [4, 4, 4], num_class)

def resnet32(num_class):
    return ResNet(BasicBlock, [5, 5, 5], num_class)

def resnet44(num_class):
    return ResNet(BasicBlock, [7, 7, 7], num_class)

def resnet56(num_class):
    return ResNet(BasicBlock, [9, 9, 9], num_class)

def test():
    net = resnet20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    