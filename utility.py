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
    parser.add_argument("--beta", help = "momemtum beta", type=float, default=0.9, required=False)
    parser.add_argument("--which_norm", help = "clipping norm type", type=int, default=2, required=False)
    parser.add_argument("--C", help = "clipping C", type=float, default=1, required=False)

    return parser.parse_args()



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
