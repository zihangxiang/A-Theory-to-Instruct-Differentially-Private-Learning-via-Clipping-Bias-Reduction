import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from PIL import Image
#
from . import dataset_setup, wrn
import datasets.RESNETS as resnet_colection
###########################################################################################
print('\n==> Using svhn data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/svhn'
print('==> dataset located at: ', data_file_root)
num_of_classes = 10

device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()
###########################################################################################
T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
transformation = T.Compose([
                            
                            T.RandomHorizontalFlip(),  
                            
                            T.ToTensor(),
                            T_normalize,
                                    
                            ])


def get_all_dataset(seed = None):
    dataset = torchvision.datasets.SVHN(
                                    root = data_file_root,
                                    split = 'train',
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
        
    dataset_test = torchvision.datasets.SVHN(
                                            data_file_root,
                                            split = 'test',
                                            download=  True,
                                            transform = T.Compose([
                                                                T.ToTensor(),
                                                                T_normalize,
                                                                ]),
                                            
                                            )   
    
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
class model(nn.Module):
    
    def __init__(self, num_of_classes):
        super().__init__()  
        self.num_of_classes = num_of_classes        
        self.my_model_block = resnet_colection.resnet20(num_of_classes)
  
    
    
    def forward(self, x):
        return self.my_model_block(x)

##################################################################################################
