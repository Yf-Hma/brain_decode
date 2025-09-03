# MIT License
# 
# Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pickle
import numpy
import torch
import os

import torch
from torch.utils.data import IterableDataset
import random


class EEG2TextIterableDataset(IterableDataset):
    def __init__(self, dataset, batch_size=1):
        super(EEG2TextIterableDataset, self).__init__()
        self.dataset = list(dataset)
        self.batch_size = batch_size

        self.indices = list(range(len(self.dataset)))
        self.index_in_epoch = 0

    def len (self):
        return len (self.indices)
    def __iter__(self):
        self.index_in_epoch = 0
        random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.index_in_epoch >= len(self.dataset):
            raise StopIteration

        batch_signals = []
        batch_texts = []
        num_items_in_batch = 0

        while num_items_in_batch < self.batch_size and self.index_in_epoch < len(self.dataset):
            idx = self.indices[self.index_in_epoch]
            item = self._get_single_item(idx)
            batch_signals.append(item['signal'].transpose(0, 1))
            batch_texts.append(item['text_output'])
            self.index_in_epoch += 1
            num_items_in_batch += 1

        if batch_signals:
            batch_signals_tensor = torch.stack(batch_signals)
            return {"signal": batch_signals_tensor, "text_output": batch_texts}
        else:
            raise StopIteration


    def _get_single_item(self, idx):
        return self.dataset[idx]


#from . import configs

def get_dataset_epoch(data_path: str, sub: str, epoch: int, type: str):

    '''
        sub: subject 
        tytpe: task type, imagine or read
    '''

    paths = f"{data_path}/derivatives/preprocessed_pkl/sub-{sub}/eeg/sub-{sub}_task-{type}_run-0{str(epoch)}_eeg.pkl"

    if not os.path.exists(paths):
        paths = f"{data_path}/derivatives/preprocessed_pkl/sub-{sub}/eeg/sub-{sub}_task-{type}_run-{str(epoch)}_eeg.pkl"
        if not os.path.exists(paths):
            print ("Path does not exists:", paths)
            exit ()

    #if not os.path.exists(paths): return list()


    
    pickles = pickle.load(open(paths, "rb"))
    #print(sub, epoch, len(pickles))

    for idx, trial in enumerate(pickles):
        assert isinstance(trial['input_features'], numpy.ndarray)
        assert trial['input_features'].dtype == numpy.float64

        if type == "imagine":
            assert trial['input_features'].shape == (1, 125, 1651)
        elif type =="read":
            assert trial['input_features'].shape == (1, 125, 2501)
            
        input_features = trial['input_features'][0, :122, :]*1000000
        mean = numpy.absolute(numpy.mean(input_features, axis=1))
        stds = numpy.std(input_features, axis=1)
        assert isinstance(input_features, numpy.ndarray)
        assert input_features.dtype == numpy.float64

        if type == "imagine":
            assert input_features.shape == (122, 1651)

        elif type =="read":
            assert input_features.shape == (122, 2501)

        assert (mean > 0).all() and (mean < 10000).all()
        assert (stds > 0).all() and (stds < 10000).all()
    return pickles

def get_dataset_subj(data_path: str, sub: str, type: str):

    data_train = []
    data_test = []
    for epoch in range(1, 46):
        pickles = get_dataset_epoch(data_path, sub=sub, epoch=epoch, type = type)
        for trial in pickles:
            input_features = trial['input_features'][0, :122, :]*1000000
            input_ids = trial['text'].strip()

            input_features = numpy.float32(input_features)
            input_features = torch.tensor(input_features)
            # dsplit["input_features"].append(input_features)
            # dsplit["labels"].append(input_ids)

            if epoch < 38:
                data_train.append ({"signal": input_features, "text_output": input_ids})
            else:
                data_test.append ({"signal": input_features, "text_output": input_ids})

    return data_train, data_test


def get_dada_loaders (data_path, subject, type, batch_size=16):
    data_train, data_test = get_dataset_subj(data_path, subject)

    train_set = EEG2TextIterableDataset(data_train, batch_size=batch_size)
    test_set = EEG2TextIterableDataset(data_test, batch_size=batch_size)

    return train_set, test_set

def get_dada_loaders_all (data_path, type, batch_size, val_batch_size):
    data_train_1, data_test_1 = get_dataset_subj(data_path, '01', type)
    data_train_2, data_test_2 = get_dataset_subj(data_path, '02', type)
    data_train_3, data_test_3 = get_dataset_subj(data_path, '03', type)
    data_train_4, data_test_4 = get_dataset_subj(data_path, '04', type)
    data_train_5, data_test_5 = get_dataset_subj(data_path, '05', type)

    data_train = data_train_1 + data_train_2 + data_train_3 + data_train_4 + data_train_5
    data_test = data_test_1 + data_test_2 + data_test_3 + data_test_4 + data_test_5

    train_set = EEG2TextIterableDataset(data_train, batch_size=batch_size)
    test_set = EEG2TextIterableDataset(data_test, batch_size=val_batch_size)

    return train_set, test_set

#####################################
# if __name__ == '__main__':

#     train_set, test_set = get_dada_loaders_all (configs.DATA_PATH, "imagine")

#     for i, item in enumerate(train_set):

#         print (item["signal"].shape)
#         print (item["text_output"])
#         break
