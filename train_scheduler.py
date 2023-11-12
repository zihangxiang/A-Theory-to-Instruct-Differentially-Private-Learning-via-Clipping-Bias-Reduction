                    
import enum
import torch
import time
from tqdm import tqdm
import random
import utility
import torchvision
import torchvision.transforms as T
from functorch import make_functional_with_buffers
from functorch import vmap, grad
from copy import deepcopy
import os
import numpy as np
import math
''' '''
import logger

''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}

class train_master:
    def __init__(self, *,
                model,
                loaders = (None, None, None),
                train_setups = dict(),
                arg_setup = None,
                ):
        self.data_logger = utility.log_master(root = arg_setup.log_dir)
        logger.init_log(dir = arg_setup.log_dir)
        self.arg_setup = arg_setup

        # self.data_recorder = logger.data_recorder(f'clip_c{self.arg_setup.C}.json')

        self.model = model   
        self.num_of_classes = self.model.num_of_classes
        
        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(self.model), disable_autograd_tracking=True)

        self.times_larger = 1 

        self.loaders = {'train': loaders[0], 'val': loaders[1], 'test': loaders[2]}
        self.train_setups = train_setups
        
        self.loss_metric = self.train_setups['loss_metric']
        
        ''' sanity check '''
        if self.loaders['train'] is None and self.loaders['val'] is None and self.loaders['test'] is None:
            raise ValueError('at least one loader must be provided')
        for setup in TRAIN_SETUP_LIST:
            if setup not in self.train_setups:
                raise ValueError(f'{setup} must be provided in train_setups')
        for setup in self.train_setups:
            if setup is None:
                raise ValueError(f'invalid setups (no NONE setup allowed): {self.train_setups}')
        
        ''' processing the model '''
        self.sigma = self.train_setups['sigma']
        logger.write_log(f'==>  sigma: {self.sigma}')
        
        ''' set the optimizer after extension '''
        self.optimizer = self.train_setups['optimizer']
        
        ''''''
        self.count_parameters() 
        print(f'==> have {torch.cuda.device_count()} cuda devices')

        self.shape_interval = []
        self.shape_list = []
        last = 0
        for p in self.model.parameters():
            if p.requires_grad:
                self.shape_list.append(p.shape)
                total_param_sub = p.numel()
                self.shape_interval.append([last, last + total_param_sub])
                last += total_param_sub
            else:
                self.shape_interval.append(None)
        self.all_indexes = list(range(self.arg_setup.usable_train_data_samples))
    
        # self.reindexing = self.get_reindex(self.num_of_models)
        
        self.grad_momentum = [ torch.zeros_like(p.data) if p.requires_grad else None for p in self.model.parameters()  ]
        self.iterator_check = [0 for _ in self.model.parameters()]
        self.per_grad_momemtum = [ 0 for _ in self.model.parameters()  ]

        self.norm_choices = [1+0.25*i for i in range(16)]
        self.avg_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        self.avg_inverse_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        
        loader = self.loaders['test']
        '''get whole test data '''
        print('==> stacking test data...')
        self.whole_data_container_test = None
        self.whole_label_container_test = None
        self.whole_index_container_test = None
        for index, train_batch in enumerate(loader):
                        # print(index, end='/')
            # if isinstance(train_batch[1], list) and len(train_batch[1]) ==2:
            #     data_index = train_batch[1][1]
            #     train_batch = (train_batch[0], train_batch[1][0])
            #     # batch_para_batch = None
                
            ''' get training data '''
            inputs, targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
            if self.whole_data_container_test is None:
                self.whole_data_container_test = inputs
                self.whole_label_container_test = targets
            else:   
                self.whole_data_container_test = torch.cat([self.whole_data_container_test, inputs], dim=0)
                self.whole_label_container_test = torch.cat([self.whole_label_container_test, targets], dim=0)
        print(f'==> test data size: {self.whole_data_container_test.size()}')
        print(f'==> all labels:', set(self.whole_label_container_test.tolist()))
  

        '''logging'''
        self.data_logger.write_log(f'weighted_recall.csv', self.arg_setup)
        logger.write_log(f'arg_setup: {self.arg_setup}')
        for i in range(torch.cuda.device_count()):
            logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=True)

    def count_parameters(self):
        total = 0
        cnn_total = 0
        linear_total = 0

        tensor_dic = {}
        for submodule in self.model.modules():
            for s in submodule.parameters():
                if s.requires_grad:
                    if id(s) not in tensor_dic:
                        tensor_dic[id(s)] = 0
                    if isinstance(submodule, torch.nn.Linear):
                            tensor_dic[id(s)] = 1

        for p in self.model.parameters():
            if p.requires_grad:
                total += int(p.numel())
                if tensor_dic[id(p)] == 0:
                    cnn_total += int(p.numel())
                if tensor_dic[id(p)] == 1:
                    linear_total += int(p.numel())

        self.cnn_total = cnn_total
        logger.write_log(f'==>  model parameter summary:')
        logger.write_log(f'     non_linear layer parameter: {self.cnn_total}' )
        self.linear_total = linear_total
        logger.write_log(f'     Linear layer parameter: {self.linear_total}' )
        self.total_params = self.arg_setup.total_para = total
        logger.write_log(f'     Total parameter: {self.total_params}\n' )
        

    def train(self):
        
        s = time.time()
        for epoch in range(self.train_setups['epoch']):
            logger.write_log(f'\n\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            ''' lr rate scheduler '''
            self.epoch = epoch
            
            train_metrics, val_metrics, test_metrics = None, None, None
            self.record_data_type = 'weighted_recall'

            
            ''' training '''
            if self.loaders['train'] is not None:
                train_metrics = self.one_epoch(train_or_val = Phase.TRAIN, loader = self.loaders['train'])
                for i in range(torch.cuda.device_count()):
                    logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=False)
            
            ''' validation '''
            if self.loaders['val'] is not None:
                val_metrics = self.one_epoch(train_or_val = Phase.VAL, loader = self.loaders['val'])

            ''' testing '''
            if self.loaders['test'] is not None:
                test_metrics = self.one_epoch(train_or_val = Phase.TEST, loader = self.loaders['test'])

            '''logging data '''
            data_str = (' '*3).join([
                                f'{epoch}',
                                f'{ float( train_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if train_metrics else 'NAN',

                                f'{ float( val_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if val_metrics else 'NAN',

                                f'{ float( test_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if test_metrics else 'NAN',
                                ])
            
            self.data_logger.write_log(f'{self.record_data_type}.csv', data_str)
        

        ''' ending '''

        # self.data_recorder.save()
        logger.write_log(f'\n\n=> TIME for ALL : {time.time()-s:.2f}  secs')
    
    def _per_sample_augmentation(self):
        ''' per sample augmentation '''
        # if self.pub_num == 0:
        return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)


    def get_per_grad(self, inputs, targets):

        ''''''
        def compute_loss(model_para, buffers,  inputs, targets):
            # print(f'inputs shape: {inputs.shape}')
            predictions = self.worker_model_func(model_para, buffers, inputs)
            # print(f'predictions shape: {predictions.shape}, targets shape: {targets.shape}')
            ''' only compute the loss of the first(private) sample '''
            predictions = predictions[:1]
            targets = targets[:1]
            
            loss = self.loss_metric(predictions, targets.flatten()) #* inputs.shape[0]
            return loss
        def self_aug_per_grad(model_para, buffers, inputs, targets):
            per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
            # for _ in range(self.arg_setup.self_aug_times):
            #     t_inputs = self.transformation(inputs)
            #     cur_grad = grad(compute_loss)(model_para, buffers, t_inputs, targets)
            #     per_grad = [p + g for p, g in zip(per_grad, cur_grad)]
            # per_grad = [p / (self.arg_setup.self_aug_times + 1) for p in per_grad]
            return per_grad
        per_grad = vmap(self_aug_per_grad, in_dims=(None, None, 0, 0), randomness='same')(self.worker_param_func, self.worker_buffers_func, inputs, targets)
        return list(per_grad)
       
        
    def one_epoch(self, *, train_or_val, loader):
        metrics = utility.ClassificationMetrics(num_classes = self.num_of_classes)
        metrics.num_images = metrics.loss = 0 
        is_training = train_or_val is Phase.TRAIN
 
        with torch.set_grad_enabled(is_training):
            self.model.train(is_training)
            s = time.time()
            if is_training: 
                print(f'==> have {len(loader)} iterations in this epoch')
                for index, train_batch in enumerate(loader):
                    ''' get training data '''
                    the_inputs, the_targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
                    
                    pub_inputs, pub_targets = self._per_sample_augmentation()
                    
                    new_inputs = torch.concat([the_inputs, pub_inputs], dim = 0)
                    new_targets = torch.concat([the_targets, pub_targets], dim = 0)
                    
                    new_inputs = torch.stack(torch.split(new_inputs, 1, dim = 0))
                    new_targets = torch.stack(torch.split(new_targets, 1, dim = 0))
                        
                    per_grad = self.get_per_grad(new_inputs, new_targets)

                    self.other_routine( per_grad )
                    
                    '''update batch metrics'''
                    with torch.no_grad():
                        predictions = self.model(the_inputs)
                        loss = self.train_setups['loss_metric']( predictions, the_targets.flatten() )
                    metrics.batch_update(loss, predictions, the_targets)

                # self.data_recorder.add_record('train_acc', float(metrics.__getattr__(self.record_data_type)))
                
                    
            else:
                for batch in loader:
                    inputs, targets = map(lambda x: x.to(self.train_setups['device']), batch)
                    
                    predicts = self.model(inputs)
                    loss = self.train_setups['loss_metric']( predicts, targets.flatten() )
                    
                    '''update batch metrics'''
                    metrics.batch_update(loss, predicts, targets)

                # self.data_recorder.add_record('test_acc', float(metrics.__getattr__(self.record_data_type)))

        metrics.loss /= metrics.num_images
        logger.write_log(f'==> TIME for {train_or_val}: {int(time.time()-s)} secs')
        logger.write_log(f'    {train_or_val}: {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type))*100:.2f}%' )
        
        return metrics  

    def clip_per_grad(self, per_grad):
        
        per_grad_norm = ( self._compute_per_grad_norm(per_grad, which_norm = self.arg_setup.which_norm) + 1e-6 )

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.arg_setup.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad):
            ''' normalizing '''
            # per_grad[index] = p / self._make_broadcastable(per_grad_norm / self.arg_setup.C, p) 
            ''' clipping '''
            # print(f'p shape: {p.shape}, mutiplier shape: {multiplier.shape}')
            per_grad[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad


        
    def other_routine(self, per_grad):
        
        ''' vanilla dp-sgd '''
        per_grad = self.clip_per_grad(per_grad)
        assert len(self.iterator_check) == len(per_grad)
        for p_stack, p in zip(per_grad, self.model.parameters()):
            if p.requires_grad:
                p.grad = torch.sum(p_stack, dim = 0) 
                p.grad += self.arg_setup.C * self.sigma * torch.randn_like(p.grad) 
                p.grad /= self.arg_setup.expected_batchsize
                
        ''' gradient momentum '''     
        for index, p in enumerate(self.model.parameters()):
            p.grad = self.arg_setup.beta * self.grad_momentum[index] + p.grad
            self.grad_momentum[index] = torch.clone(p.grad)
                     
        self.model_update()
        
    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)
        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
    def model_update(self):
        ''' update the model '''
        self.optimizer.step()
        
        ''' copy global model to worker model'''
        for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
            assert p_worker.shape == p_model.data.shape
            p_worker.copy_(p_model.data)
            

    def flatten_to_rows(self, leading_dim, iterator):
        return torch.cat([p.reshape(leading_dim, -1) for p in iterator], dim = 1)
    