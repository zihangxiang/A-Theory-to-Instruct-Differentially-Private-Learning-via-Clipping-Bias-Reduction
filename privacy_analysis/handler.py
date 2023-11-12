import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
#
class MyBatchSampler(torch.utils.data.Sampler):
    def __init__(self,*, max_iter_num, expected_batch_size, index_choices, not_reset_counter = False, enable_pois_sampling = False):
        
        
        self.max_iter_num = max_iter_num
        # print('max_iter_num', max_iter_num)
        self.expected_batch_size = expected_batch_size
        ''' rand per? '''
        self.original_index = [i for i in index_choices]
        self.index_choices = np.random.permutation(index_choices)
        
        self.n = len(index_choices)
        self.not_reset_counter = not_reset_counter
        
        self.enable_pois_sampling = enable_pois_sampling
        
        print(id(self), 'contains', len(self.index_choices), 'samples')
        
    def __iter__(self):
        # print('\n\n==> reinitialize MyBatchSampler...')
        
        self.index_choices = np.random.permutation(self.original_index)
        self.indexes = []
        self.iter_counter = 0
        while True:
            if self.iter_counter >= self.max_iter_num:
                # self.reset_index_choices()
                break
            
            if len(self.index_choices) == 0:
                self.reset_index_choices()
                
            
            ''' poisson sampling ? '''
            if not self.enable_pois_sampling:
                number = self.expected_batch_size
            else:
                number = np.random.binomial(self.n, self.expected_batch_size/self.n)
                
    
            ''' iteration dependent [1,2,3] [4,5] [6,7,8,9]...'''
            to_be_returned = self.index_choices[:number]
            self.indexes.append( to_be_returned )
            self.index_choices = self.index_choices[number:]
            
            self.iter_counter += 1
        return iter(self.indexes)
                
    def reset_index_choices(self):
        self.original_index = np.random.permutation(self.original_index)
        self.index_choices = [i for i in self.original_index]
        if self.not_reset_counter:
            return
        self.iter_counter = 0
                 
    def __len__(self):
        return self.max_iter_num

def to_special_dataloader( *, loader, sampling_rate, batch_para_computer_len, batch_para_computer_batch_size): 
    assert batch_para_computer_len >= batch_para_computer_batch_size
    assert len(loader.dataset) >= batch_para_computer_len
    # all_indexes = [i for i in range(len(loader.dataset))]
    return  DataLoader(
                        dataset = loader.dataset,
                        batch_sampler = MyBatchSampler(
                            max_iter_num = int(1/sampling_rate), 
                            expected_batch_size = int(sampling_rate * (len(loader.dataset) - batch_para_computer_len)), 
                            index_choices = list(range(len(loader.dataset) - batch_para_computer_len)),
                            enable_pois_sampling = False,
                            ),
                        num_workers = 4,
                        pin_memory = loader.pin_memory,
                        ), \
            None
            # DataLoader(
            #             dataset = loader.dataset,
            #             batch_sampler = MyBatchSampler(
            #                 max_iter_num = int(1/sampling_rate),
            #                 expected_batch_size = batch_para_computer_batch_size, 
            #                 index_choices = list(range(len(loader.dataset) - batch_para_computer_len, len(loader.dataset))),
            #                 not_reset_counter = True,
            #                 enable_pois_sampling = False,
            #                 ),
            #             num_workers = 4,
            #             pin_memory = loader.pin_memory,
            #             )
    




class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def privatized_loader(train_dataset, expected_batchsize):
    # '''poisson sampling'''
    # return DataLoader(
    #     dataset = train_dataset,
    #     batch_sampler = PoissonSampler(len(train_dataset), expected_batchsize),
    #     num_workers = 4,
    #     pin_memory = True,
    # )

    ''' normal loader '''
    return DataLoader(
                    dataset = train_dataset,
                    batch_size = expected_batchsize,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True,
                    drop_last = False,
                    )