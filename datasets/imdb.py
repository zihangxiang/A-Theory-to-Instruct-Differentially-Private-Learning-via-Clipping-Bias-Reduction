import torch
import torch.nn as nn
import torchvision
import torchtext
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import os
import torch.nn.functional as F

#
from . import dataset_setup
###########################################################################################
print('\n==> Using imdb data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/IMDB'
print('==> dataset located at: ', data_file_root)
num_of_classes = 2

device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()
###########################################################################################
global vocab_size
vocab_size = None

SEQ_LENGTH = 128

def get_data():
    files = ['train_data.pt', 'test_data.pt',  'train_labels.pt', 'test_labels.pt', 'vocab.pt']
    the_path = str( Path(__file__).parent) + '/nltk'
    file_we_have = os.listdir(the_path)
    # print('==> we have files: ', file_we_have)

    files_ready = True
    for filename in files:
        if filename not in file_we_have:
            files_ready = False
            break

    if files_ready:
        print('==> we have all files, loading from disk')
        train_data = torch.load(f'{the_path}/train_data.pt')
        test_data = torch.load(f'{the_path}/test_data.pt')
        train_labels = torch.load(f'{the_path}/train_labels.pt')
        test_labels = torch.load(f'{the_path}/test_labels.pt')
        vocab = torch.load(f'{the_path}/vocab.pt')
    else:
        print('==> we do not have the files, we need to form them')
        from torchtext.vocab import build_vocab_from_iterator
        # text processing
        import re
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem.wordnet import WordNetLemmatizer
        nltk.download('stopwords')#, download_dir = '/nltk')
        nltk.download('wordnet')#, download_dir = '/nltk')
        stopwords = set(stopwords.words('english'))

        ''' functions '''
        def rm_link(text):
            return re.sub(r'https?://\S+|www\.\S+', '', text)

    
        def rm_punct2(text):
            return re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)

        def rm_html(text):
            return re.sub(r'<[^>]+>', '', text)

        def space_bt_punct(text):
            pattern = r'([.,!?-])'
            s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
            s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
            return s

        def rm_number(text):
            return re.sub(r'\d+', '', text)

        def rm_whitespaces(text):
            return re.sub(r' +', ' ', text)

        def rm_nonascii(text):
            return re.sub(r'[^\x00-\x7f]', r'', text)

        def rm_emoji(text):
            emojis = re.compile(
                                '['
                                u'\U0001F600-\U0001F64F'  # emoticons
                                u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                                u'\U0001F680-\U0001F6FF'  # transport & map symbols
                                u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                                u'\U00002702-\U000027B0'
                                u'\U000024C2-\U0001F251'
                                ']+',
                                flags = re.UNICODE
                            )
            return emojis.sub(r'', text)

        def spell_correction(text):
            return re.sub(r'(.)\1+', r'\1\1', text)

        def clean_pipeline(text):    
            no_link = rm_link(text)
            no_html = rm_html(no_link)
            space_punct = space_bt_punct(no_html)
            no_punct = rm_punct2(space_punct)
            no_number = rm_number(no_punct)
            no_whitespaces = rm_whitespaces(no_number)
            no_nonasci = rm_nonascii(no_whitespaces)
            no_emoji = rm_emoji(no_nonasci)
            spell_corrected = spell_correction(no_emoji)
            return spell_corrected


        def form_data(dataset_train, dataset_test):

            dataholder_train_data = []
            dataholder_train_label = []

            dataholder_test_data = []
            dataholder_test_label = []
            def get_vocab():
                for index, (label, line) in enumerate(dataset_train):
                    cleaned_line = clean_pipeline(line)
                    # print(index, label, cleaned_line.split())
                    # cleaned_line = preprocess_pipeline(cleaned_line)
                    spl = cleaned_line.split()
                    dataholder_train_data.append( spl )
                    dataholder_train_label.append( label-1 )
                    yield spl
                for index, (label, line) in enumerate(dataset_test):
                    cleaned_line = clean_pipeline(line)
                    # print(index, label, cleaned_line.split())
                    # cleaned_line = preprocess_pipeline(cleaned_line)
                    spl = cleaned_line.split()
                    dataholder_test_data.append( spl )
                    dataholder_test_label.append( label-1 )
                    yield spl
            vocab = build_vocab_from_iterator(get_vocab(), specials=["<unk>"])
            vocab.append_token("<pad>")

            # print(11111, dataholder_train_data[23])
            dataholder_train_id = []
            for line in dataholder_train_data:
                dataholder_train_id.append( [vocab[token] for token in line] )

            dataholder_test_id = []
            for line in dataholder_test_data:
                dataholder_test_id.append( [vocab[token] for token in line] )

            import numpy as np
            def pad_features(reviews, pad_id, seq_length=128):
                # features = np.zeros((len(reviews), seq_length), dtype=int)
                features = np.full((len(reviews), seq_length), pad_id, dtype=int)
                for i, row in enumerate(reviews):
                    # if seq_length < len(row) then review will be trimmed
                    features[i, :len(row)] = np.array(row)[:seq_length]

                return features

            seq_length = SEQ_LENGTH
            dataholder_train_feature = pad_features(dataholder_train_id, pad_id = vocab['<pad>'], seq_length = seq_length) 
            dataholder_test_feature = pad_features(dataholder_test_id, pad_id = vocab['<pad>'], seq_length = seq_length)

            return torch.tensor(dataholder_train_feature), \
                    torch.tensor( dataholder_train_label), \
                    torch.tensor(dataholder_test_feature), \
                    torch.tensor( dataholder_test_label),  \
                    vocab

        dataset_train = torchtext.datasets.IMDB(root=data_file_root, split='train')
        dataset_test = torchtext.datasets.IMDB(root=data_file_root, split='test')

        print(f'forming data...')
        train_data, train_labels, test_data, test_labels, vocab = form_data(dataset_train, dataset_test)

        print('saving data...')
        torch.save(train_data, f'{the_path}/train_data.pt')
        torch.save(train_labels, f'{the_path}/train_labels.pt')
        torch.save(test_data, f'{the_path}/test_data.pt')
        torch.save(test_labels, f'{the_path}/test_labels.pt')
        torch.save(vocab, f'{the_path}/vocab.pt')

    return  [train_data, train_labels], [test_data, test_labels], vocab

def get_all_dataset(seed=0):
    from torch.utils.data import TensorDataset

    dataset_train, dataset_test, vocab = get_data()
    global vocab_size
    vocab_size = len(vocab)
    dict_vocab = torchtext.vocab.GloVe(name='6B', dim=100, cache = str(data_file_root)+'/glove')

    class myDataset(torch.utils.data.Dataset):
        def __init__(self, datas, labels):
            super().__init__()
            self.datas = datas
            self.labels = labels
            # print(111, len(self.datas), len(self.labels))
        def __len__(self):
            return len(self.datas)

        def __getitem__(self, idx):
            token_ids = self.datas[idx].tolist()
            # print(12121212, token_ids, len(token_ids))

            token_str = vocab.lookup_tokens(token_ids)
            # print(2222222, token_str,   len(token_str))
            token_vec = dict_vocab.get_vecs_by_tokens(token_str)
            return token_vec, self.labels[idx]

    return myDataset(dataset_train[0],dataset_train[1]), None,  myDataset(dataset_test[0],dataset_test[1])


def get_all(batchsize_train = None, seed = None):
    dataset_train, dataset_val, dataset_test = get_all_dataset(seed = seed)

    print(f'==> dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}')

    # print(111, dataset_train[0][0].shape)

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
                                batch_size = 1000,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_val, dataset_test), (dataloader_train, None, dataloader_test)
    
'''model setup'''
##################################################################################################
class model(nn.Module):
    def __init__(self,  num_of_classes, hidden_size = 64, dropout = 0):
        super(model, self).__init__()
        self.num_of_classes = num_of_classes
        self.max_length, self.ebd_size = 100, 100

        self.fc = nn.Linear(hidden_size, num_of_classes)
        
        self.cell_linear_in = nn.Linear(self.ebd_size, hidden_size*3)
        self.cell_linear_hidden = nn.Linear(hidden_size, hidden_size*3)
        self.hidden_size = hidden_size

    def forward(self, x):
        print(111, x.shape)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) != 3:
            raise ValueError('input shape should be 3D ')

        hid_i = torch.zeros(1, self.hidden_size ).to(device)
        hid_i_back = torch.zeros(1, self.hidden_size ).to(device)

        ''' no norm '''
        NLA_1 = torch.sigmoid
        NLA_2 = torch.tanh
        for l in range(x.shape[1]):
            if l > self.max_length:
                break
            tmp_i = self.cell_linear_in(x[:,l,:])
            tmp_h = self.cell_linear_hidden(hid_i)
            
            r_i, z_i, n_i = torch.chunk(tmp_i, 3, dim=1)
            r_h, z_h, n_h = torch.chunk(tmp_h, 3, dim=1)

            r_t = NLA_1(r_i + r_h)
            z_t = NLA_1(z_i + z_h)
            n_t = NLA_2(n_i + r_t * n_h)
            hid_i = (1 - z_t) * n_t + z_t * hid_i
        out = self.fc(hid_i)


        ''' vector norm '''
        NLA_1 = torch.sigmoid
        NLA_2 = torch.tanh
        for l in range(x.shape[1]):
            if l > self.max_length:
                break
            tmp_i = self.cell_linear_in(x[:,l,:])
            tmp_h = self.cell_linear_hidden(hid_i)
            
            tmp_i = (tmp_i - tmp_i.mean(dim=1, keepdim=True)) / tmp_i.std(dim=1, keepdim=True)
            tmp_h = (tmp_h - tmp_h.mean(dim=1, keepdim=True)) / tmp_h.std(dim=1, keepdim=True)

            r_i, z_i, n_i = torch.chunk(tmp_i, 3, dim=1)
            r_h, z_h, n_h = torch.chunk(tmp_h, 3, dim=1)

            r_t = NLA_1(r_i + r_h)
            z_t = NLA_1(z_i + z_h)
            n_t = NLA_2(n_i + r_t * n_h)
            hid_i = (1 - z_t) * n_t + z_t * hid_i
        out = self.fc(hid_i)

        return out

##################################################################################################

def main():

    (dataset_train, dataset_val, dataset_test), (dataloader_train, _, dataloader_test) = get_all()
    for data, label in dataloader_train:
        # print(data, label)
        print(data.shape, label.shape)
        break
    the_model = model(num_of_classes = 2)

    outputs = the_model(data)
    print(111, outputs.shape)


