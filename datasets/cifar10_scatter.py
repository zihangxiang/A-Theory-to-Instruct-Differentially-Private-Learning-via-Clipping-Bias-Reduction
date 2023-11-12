import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader
from pathlib import Path
from PIL import Image
from functools import partial
from kymatio.torch import Scattering2D

from . import norm_layer as nl

def get_scatter_transform():
    shape = (32,32,3)
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)
scattering, K, _ = get_scatter_transform()



#
from . import dataset_setup, wrn
import datasets.RESNETS as resnet_colection
###########################################################################################
print('\n==> Using cifar10 data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/cifar10'
print('==> dataset located at: ', data_file_root)
num_of_classes = 10
name = 'cifar10_scatter'
device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()

###########################################################################################
T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )


transformation = T.Compose([
                            
                            # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                            # T.RandomCrop(size=(32, 32), padding=4),
                            T.RandomHorizontalFlip(),  
                            # T.RandomRotation(degrees=(-10, 10),),
                            
                            T.ToTensor(),
                            T_normalize,
                            
                            
                            # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                            # T.RandomPerspective(distortion_scale=0.5, p=1.0),
                            # T.RandomHorizontalFlip(),
                            ])


def get_all_dataset(seed = None):
    dataset = torchvision.datasets.CIFAR10(
                                    root = data_file_root,
                                    train = True,
                                    download = True,
                                    transform = transformation,
                                    )
    
    if seed is not None:
        dataset_train, dataset_val = random_split(
                                                    dataset, 
                                                    [len(dataset) - 0, 0],
                                                    generator=torch.Generator().manual_seed(seed)
                                                )
    else:
        dataset_train, dataset_val = random_split(dataset, [len(dataset) - 1, 1])
        
    dataset_test = torchvision.datasets.CIFAR10(
                                            data_file_root,
                                            train = False,
                                            download=  True,
                                            transform = T.Compose([
                                                                T.ToTensor(),
                                                                T_normalize,
                                                                ]),
                                            
                                            )   
    
    # dataset_train.__getitem__ = __getitem___special.__get__(dataset_train) 
    return dataset_train, dataset_val, dataset_test


def get_all(batchsize_train = 128, seed = None,):
    dataset_train, dataset_val, dataset_test = get_all_dataset(seed = seed)

    # training loader
    dataloader_train = DataLoader(
                                dataset = dataset_train,
                                batch_size = batchsize_train,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    # # validation loader
    # dataloader_val = DataLoader(
    #                             dataset = dataset_val,
    #                             batch_size = 512,
    #                             shuffle = True,
    #                             num_workers = 4,
    #                             pin_memory = (device.type == 'cuda'),
    #                             drop_last = False,
    #                             )
    # testing loader
    dataloader_test = DataLoader(
                                dataset = dataset_test,
                                batch_size = 500,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_val, dataset_test), (dataloader_train, None, dataloader_test)
    
'''model setup'''
##################################################################################################
# class model(nn.Module):
    
#     def __init__(self, num_of_classes):
#         super().__init__()  
#         self.num_of_classes = num_of_classes
#         # self.my_model_block = resnet_colection.resnet20()
        
#         self.my_model_block = resnet_colection.resnet20(num_of_classes)
#         # self.my_model_block = wrn.WideResNet(
#         #                                     depth = 16, 
#         #                                     num_classes = num_of_classes, 
#         #                                     widen_factor = 4, 
#         #                                     dropRate = 0.0,
#         #                                     )
    
    
#     def forward(self, x):
#         return self.my_model_block(x)

# conv_layer = partial(nn.Conv2d, bias = False)
conv_layer = partial(nl.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)


class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.conv_tras = nn.Conv2d(48, 243, kernel_size=1, stride=1, padding=0, bias=False)
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                raise
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        # act = nn.Tanh
        act = nn.ELU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = conv_layer(c, v)

                layers += [conv2d, act()]

                layers += [nn.GroupNorm(4, v, affine=False)]

                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        # print(111, x.shape)
        if self.in_channels != 3:
            x = self.norm(x.view(x.shape[0], -1, 8, 8))
        if x.shape[1] != 243:
            x = self.conv_tras(x)
        # print(22, x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # self.conv1

        # x = x.view(x.shape[0], -1, 8, 8)
        # out = self.bn1( F.elu( self.conv1(x) ) )
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.adaptive_avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)

        return x

print(f'K   : {K}')
model = CIFAR10_CNN(K, input_norm="GroupNorm", num_groups = 1, bn_stats = None, size = None)

model.device = device
model.num_of_classes = num_of_classes

#  model = CNNS[dataset](K, input_norm=input_norm, num_groups=num_groups, size=size)

##################################################################################################
