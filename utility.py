import torch
import os
import time
import torchvision.transforms as T
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class ClassificationMetrics:
    """Accumulate per-class confusion matrices for a classification task."""
    metrics = ('accur', 'recall', 'specif', 'precis', 'npv', 'f1_s', 'iou')

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = self.fn = self.fp = self.tn = 0
        self.hit_count = 0
        self.hit_accuracy = 0
        self.num_of_prediction = 0

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        return self.tp + self.fn

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.tp + self.fn
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        # return self.count.sum()
        return ( self.tp + self.fn ).sum()

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        pred, true = pred.flatten(), true.flatten()
        classes = torch.arange(0, self.num_classes, device=true.device)
        valid = (0 <= true) & (true < self.num_classes)
        '''
        this trick:
        pred_pos is n * 1 tensor, pred is 1 * n tensor
        '''
        pred_pos = classes.view(-1, 1) == pred[valid].view(1, -1)#size(3) compare with size(110)->(3,110)
        positive = classes.view(-1, 1) == true[valid].view(1, -1)
        pred_neg, negative = ~pred_pos, ~positive
        self.tp += (pred_pos & positive).sum(dim=1)
        self.fp += (pred_pos & negative).sum(dim=1)
        self.fn += (pred_neg & positive).sum(dim=1)
        self.tn += (pred_neg & negative).sum(dim=1)
        
        self.hit_count += (pred == true).sum().item()
        self.num_of_prediction += int(pred.numel())
        
        #self.hit_accuracy = self.hit_count / ( self.tp + self.fn ).sum()
        self.hit_accuracy = self.hit_count / self.num_of_prediction
    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = self.fn = self.fp = self.tn = 0

    @property
    def accur(self):
        """Get the per-class accuracy."""
        # we avoid dividing by zero using: max(denominator, 1)
        return (self.tp + self.tn) / self.total.clamp(min=1)
    

    @property
    def recall(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fn).clamp(min=1)
    
    @property
    def specif(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fp).clamp(min=1)
    
    @property
    def npv(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fn).clamp(min=1)

    @property
    def precis(self):
        """Get the per-class precision."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def f1_s(self):  # a.k.a. Sorensen–Dice Coefficient
        """Get the per-class F1 score."""
        # we avoid dividing by zero using: max(denominator, 1)
        tp2 = 2 * self.tp
        return tp2 / (tp2 + self.fp + self.fn).clamp(min=1)

    @property
    def iou(self):
        """Get the per-class intersection over union."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def weighted(self, scores):
        """Compute the weighted sum of per-class metrics."""
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
         """Quick hack to add mean and weighted properties."""
         if name.startswith('mean_') or not name.startswith(
             'mean_') and name.startswith('weighted_'):
              metric = getattr(self, '_'.join(name.split('_')[1:]))
              return metric.mean() if name.startswith('mean_') else self.weighted(metric)
         raise AttributeError(name)

    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = torch.stack([getattr(self, m) for m in self.metrics])
        perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
        out = 'Class'.ljust(6) + ''.join(map(lambda x: x.ljust(8), self.metrics))

        if self.num_classes > 20:
            return self._total_summary(metrics, perc)

        # main stat
        out += '\n' + '-' * 60
        for i, values in enumerate(metrics.t()):
            out += '\n' + str(i).ljust(6)
            out += ''.join(map(lambda x: perc(x.mean()), values))

        return out + self._total_summary(metrics, perc)

    def _total_summary(self, metrics, perc):
        out = ''
        out += '\n' + '-' * 60

        out += '\n'+'Mean'.ljust(6)
        out += ''.join(map(lambda x: perc(x.mean()), metrics))

        out += '\n'+'Wted'.ljust(6)
        out += ''.join(map(lambda x: perc(self.weighted(x)), metrics))
        out += '\n' + 'hit accuracy: ' + f'{float(self.hit_accuracy) * 100:.2f}%'
        return out

    def disp(self, with_detail = True):
        if with_detail:
            print( self )
        else:
            metrics = torch.stack([getattr(self, m) for m in self.metrics])
            perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
            print(self._total_summary(metrics, perc))
            
    def batch_update(self, loss, logits, targets):
        self.num_images += logits.shape[0]
        self.loss += loss.item() * logits.shape[0]
        self.update(logits.data.argmax(dim=1), targets.flatten())


class log_master:
    def __init__(self, root = 'logs', ):
        self.root = f'{os.getcwd()}/{root}'
        if root not in os.listdir(os.getcwd()):
            os.makedirs(self.root, exist_ok = False)
        self.filename_existed = set()

    def write_log(self, filename, content):
        if filename not in self.filename_existed:
            self.filename_existed.add(filename)
            with open(f'{self.root}/{filename}', 'a') as file:
                file.write('\n')
                time_stamp = time.strftime('[%d_%H_%M_%S]',time.localtime(time.time())) 
                file.write(f'{time_stamp} => NEW\n')
        
        with open(f'{self.root}/{filename}', 'a') as file:
            if isinstance(content, list):
                for item in content:
                    file.write(f'{item}\n')
            else:
                file.write(f'{content}\n')
                
                


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--expected_batchsize", help = "expected_batchsize", type=int, default=1000, required=True)
    parser.add_argument("--EPOCH", help = "EPOCH", type=int, default=50, required=True)
    parser.add_argument("--epsilon", help = "the epsilon value", type=float, default=8.0, required=True)
    parser.add_argument("--lr", help = "learing rate", type=float, default=0.01, required=True)
    parser.add_argument("--log_dir", help = "log directory", type=str, default='logs', required=True)
    return parser.parse_args()



# def grad_summary(epoch, per_grad):

#     data_collection = []
#     for p in per_grad:
#         batch_size = p.shape[0]
#         data_collection.append(p.reshape(batch_size,-1))
        
#     data_collection = torch.cat(data_collection, dim = 1)
#     print(f'==> data_collection.shape: {data_collection.shape}')
    
#     samples = 15
#     selected_indexes = torch.randint(0, data_collection.shape[0], (samples,))
#     plot_data = data_collection[selected_indexes]
    
#     plt.figure(figsize=(12, 8))
#     plt.style.use('seaborn-dark')
    
#     mode_color_his = '#f97306'
#     mode_color_line = 'b'
#     num_bins = 500
#     for i in range(samples):
#         plt.subplot(4,4,i+1)
#         data = plot_data[i].cpu().numpy()
#         plt.hist(data, bins = num_bins, density = True, color = mode_color_his)
        
#         results = data
#         mu = np.mean(results)
#         sigma = np.std(results)
#         vec_norm = np.linalg.norm(results, ord = 2)
#         x = np.linspace(mu - 3*sigma, mu + 3*sigma, num_bins)
#         plt.plot(x, stats.norm.pdf(x, mu, sigma), '--', label = f'norm:{vec_norm:.1f}\n$\mu:{mu:.2f}$\n$\sigma:{sigma:.2f}$', color = mode_color_line)
        
#         # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#         plt.legend()
#         plt.title(f'grad_{i}', fontsize = 8)
    
    
#     plt.subplot(4, 4, samples+1)
#     for i in range(samples):
#         data = plot_data[i].cpu().numpy()
#         plt.hist(data, bins = num_bins, density = True)
#     plt.title(f'all grads', fontsize = 4)
    
    
#     # plt.title('Histogram of coordinate value')
#     plt.tight_layout()
#     plt.savefig(f'cor_dis_clip_{epoch}' + '.pdf')
#     exit()
    
    
# def per_grad_value(epoch, per_grad, C, th):

#     data_collection = []
#     for p in per_grad:
#         batch_size = p.shape[0]
#         data_collection.append(p.reshape(batch_size,-1))
#     data_collection = torch.cat(data_collection, dim = 1)
#     print(f'==> get per grad cordinate value summary')
#     # print(f'==> data_collection.shape: {data_collection.shape}')
    
#     total = data_collection.shape[1]
#     samples = 5
#     selected_indexes = torch.randint(0, data_collection.shape[0], (samples,))
#     plot_data = data_collection[selected_indexes]
    
#     print(f'threshold list : {[C * th * scale for scale in range(1, 11)]}')
#     print(f'C * infi_th = {C}*{th}={C*th}')
#     for i in range(samples):
#         data = abs(plot_data[i])
#         print(f'grad_{i}: ', end='')
#         for scale in range(1, 11):
#             th_ = C * th * scale
#             last_th_ = C * th * (scale - 1)
#             tmp = data[ torch.bitwise_and(data > last_th_, data <= th_) ]
#             print(f'{tmp.numel()/total:.2f}', end=', ')
#         print()
        
    
#     for i in range(samples):
#         data = torch.norm(plot_data[i], p=1.5)
#         print(f'grad_{i} l1.5 norm: {data}')
#     for i in range(samples):
#         data = torch.norm(plot_data[i], p=2)
#         print(f'grad_{i} l2 norm: {data}')
    
    
    
    

    
#     '''plot'''
#     # # plot_data = data_collection[selected_indexes][:, :100]
#     # plot_data, _ = torch.sort(abs(plot_data), dim = 1)
    
#     # plt.figure(figsize=(12, 8))
#     # plt.style.use('seaborn-dark')
    
#     # mode_color_line = 'b'
    
#     # th_ = C*th
#     # dummy = [th_] * int(plot_data.shape[1])
#     # for i in range(samples):
#     #     plt.subplot(4,4,i+1)
#     #     data = plot_data[i].cpu().numpy()
#     #     plt.plot(data, "*", markersize = 0.2)
#     #     plt.plot(dummy, label = f'C * infi_th = {C}*{th}={th_}', color = mode_color_line)

#     #     plt.legend()
#     #     plt.title(f'grad_{i}', fontsize = 8)
    
#     # plt.tight_layout()
#     # plt.savefig(f'cor_value_{epoch}' + '.pdf')
    
#     # exit()
    
    
    
# def grad_norm_summary(lr, total_epoch, average_of_dif_norm, average_of_inverse_dif_norm):
    
#     keys = sorted(list(average_of_dif_norm.keys()))
#     if len(average_of_dif_norm[keys[0]]) == 0:
#         return
#     print(f'==> making summary of the average of dif p norm')
    
    
    
#     plt.figure(figsize=(12, 8))
#     plt.style.use('seaborn-dark')
#     for i in range(len(keys)):
#         p_norm = keys[i]
#         data = average_of_dif_norm[p_norm]
#         plt.subplot(4,4,i+1)
#         plt.plot(data, label = f'p_norm:{p_norm}', linewidth = 0.8)
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'p_norms_{total_epoch}_{lr}' + '.pdf')
    
#     plt.figure(figsize=(12, 8))
#     plt.style.use('seaborn-dark')
#     for i in range(len(keys)):
#         p_norm = keys[i]
#         data = average_of_inverse_dif_norm[p_norm]
#         plt.subplot(4,4,i+1)
#         plt.plot(data, label = f'p_norm:{p_norm}', linewidth = 0.8)
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'p_norms_inverse_{total_epoch}_{lr}' + '.pdf')
    
    
    
    
# def per_grad_th_binary_summary(flattened_per_grad, thresholds):
#     # for th in thresholds:
#     #     tmp = abs(torch.clone(flattened_per_grad))
#     #     small_loc = tmp < th
#     #     big_loc = tmp >= th
        
#     #     tmp[small_loc] = 0
#     #     tmp[big_loc] = 1
        
#     #     result = tmp.sum(dim = 0).cpu().numpy()

#     #     plt.figure(figsize=(8, 4))
#     #     plt.style.use('seaborn-dark')
#     #     plt.plot(result, linewidth = 0.8)
#     #     plt.title(f'bigger than {th} is 1 and smaller than {th} is 0')
#     #     plt.tight_layout()
#     #     plt.savefig(f'BS_th_{th}' + '.pdf')
        
#     # for th in thresholds:
#     # for percent in [0.0001, 0.001, 0.01, 0.1]:
#     #     tmp = abs(torch.clone(flattened_per_grad))
#     #     s,_ =torch.topk(tmp, dim = 1, k = int(tmp.shape[1] * percent) )
        
#     #     the_one = s[:,-1:]
        
#     #     small_loc = tmp < the_one
#     #     big_loc = tmp >= the_one
        
#     #     tmp[small_loc] = 0
#     #     tmp[big_loc] = 1
        
#     #     result = tmp.sum(dim = 0).cpu().numpy()

#     #     plt.figure(figsize=(8, 4))
#     #     plt.style.use('seaborn-dark')
#     #     plt.plot(result, linewidth = 0.8)
#     #     plt.title(f'top {percent} of total is 1 and smaller is 0')
#     #     plt.tight_layout()
#     #     plt.savefig(f'BS_inner_{percent}' + '.pdf')
    
#     '''seperate l2 norm '''
#     percent = 0.1
#     tmp = abs(torch.clone(flattened_per_grad))
#     s,_ =torch.topk(tmp, dim = 1, k = int(tmp.shape[1] * percent) )
    
#     the_one = s[:,-1:]
    
#     small_loc = tmp < the_one
#     big_loc = tmp >= the_one
    
#     all_small = torch.clone(tmp)
#     all_bigger = torch.clone(tmp)
#     all_small[big_loc] = 0
#     all_bigger[small_loc] = 0
    
#     all_small = all_small.norm(dim = 1)
#     all_bigger = all_bigger.norm(dim = 1)
    
#     all_norm = tmp.norm(dim=1)
    
#     checking_num = 30
#     print(f'top {percent}')
#     for i in range(checking_num):
#         print(f'==> grad {i} all_small: {all_small[i]:.2f} all_bigger: {all_bigger[i]:.2f} all_norm: {all_norm[i]:.2f}')
#         print(f'    small portion: {all_small[i]/all_norm[i]:.2f} bigger portion: {all_bigger[i]/all_norm[i]:.2f} ')
#         print()
    
#     exit()


import json
'''json'''
def get_data_from_record(filename):
    path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    data = get_data_from_record('sgd.json')


    data_dict = {
        'sampling_noise': data['sampling_noise'],
        'last_time_norm_of_grad': data['last_time_norm_of_grad'],
        'train_acc': data['train_acc'],
        'test_acc': data['test_acc'],
    }


    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.style.use('seaborn-dark')
    
    for index, item in enumerate(data_dict):
        plt.subplot(2,2,index+1)
        plt.plot(data_dict[item], label = item, linewidth = 0.8)
        # plt.title(item)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'sampling_noise' + '.pdf')
