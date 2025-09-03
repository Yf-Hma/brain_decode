import json
import os
import sys
import re
import numpy
import torch
from torch.utils.data import IterableDataset
import random


sys.path.insert(0, os.getcwd())

import configs.configs_perceived as configs

def pre_caption(caption, max_words=200):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[: max_words])

    return caption


def read_signal (signal_path):
    bold_path = numpy.load (signal_path)
    bold_signal = torch.Tensor (bold_path)#.unsqueeze (0)
    if len (bold_signal.shape) == 3:
        padded = torch.zeros(bold_signal.shape[0], bold_signal.shape[1], configs.src_fmri_features_max - bold_signal.shape[2])
        bold_signal = torch.cat([bold_signal,padded], dim = 2)
    elif len(bold_signal.shape) == 2:
        padded = torch.zeros(bold_signal.shape[0], configs.src_fmri_features_max - bold_signal.shape[1])
        bold_signal = torch.cat([bold_signal,padded], dim = 1)
        
    return bold_signal


def make_sample(item):
    sample = {
        "text_output": item["text_output"],
        "signal": read_signal(item["bold_signal"]),
        "chunk_number": item.get("chunk_number", -1)  # handle missing "chunk"
    }
    return sample



class IndexableDataset(IterableDataset):
    def __init__(self, data, world_size=1, rank=0, transform=None,
                 shuffle=False, batch_size=None, drop_last=False, seed=None):

        self.data = data
        self.world_size = world_size
        self.rank = rank
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_mode = True
        self.drop_last = drop_last
        self.seed = seed

        # Prepare indexed version of the dataset
        #assert len(data["text_output"]) == len(data["signal"]), "Text and signal must be same length"
        self.size = len(data)


    def __iter__(self):
        # Generate index list
        indices = list(range(self.size))

        # Optional shuffle
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        # Shard the indices for DDP
        indices = indices[self.rank::self.world_size]

        # Prepare examples
        examples = []
        for idx in indices:
            examples.append(self.data[idx])

        # Return as individual samples or batches
        if self.batch_mode:
            return self._batch_iter(examples, self.batch_size)
        else:
            return iter(examples)

    def _batch_iter(self, data, batch_size):
        batch = []

        batch = {"text_output":[], "signal": [], "chunk_number": []}

        for item in data:
            sample = make_sample(item)

            batch["text_output"].append(sample["text_output"])
            batch["signal"].append(sample["signal"])
            batch["chunk_number"].append(sample["chunk_number"])

            if len(batch["text_output"]) == batch_size:
                batch["signal"] = torch.stack(batch["signal"])
                yield batch

                batch = {"text_output":[], "signal": [], "chunk_number": []}

        if len (batch["text_output"]) > 0:
            if  len (batch["text_output"]) == self.batch_size:
                batch["signal"] = torch.stack(batch["signal"])
                yield batch
            
            if  self.drop_last==False:
                batch["signal"] = torch.stack(batch["signal"])
                yield batch


def get_loaders (subject, batch_size=32, val_batch_size=32):

    with open("%s/%s/train.json"%(configs.DATA_JSON_PATH,subject)) as json_file:
        train_loader = IndexableDataset(data = json.load(json_file), batch_size=batch_size,  drop_last=True, shuffle=True)

    with open("%s/%s/test_perceived_speech_wheretheressmoke.json"%(configs.DATA_JSON_PATH,subject)) as json_file:        
        test_loader  = IndexableDataset(data = json.load(json_file), batch_size=val_batch_size, shuffle=False)

    return train_loader, test_loader


def data_builder_from_file (filename, batch_size=32):
    with open(filename) as json_file:
        data_loader  = IndexableDataset(data = json.load(json_file), batch_size=batch_size, shuffle=False)
    return data_loader


def get_loaders_all (batch_size=32, val_batch_size=32):

    with open("%s/S1/train.json"%(configs.DATA_JSON_PATH)) as json_file:
        json_s1 = json.load(json_file)

    with open("%s/S2/train.json"%(configs.DATA_JSON_PATH)) as json_file:
        json_s2 = json.load(json_file)

    with open("%s/S3/train.json"%(configs.DATA_JSON_PATH)) as json_file:
        json_s3 = json.load(json_file)

    train_loader = IndexableDataset(data = json_s1 + json_s2 + json_s3, batch_size=batch_size, drop_last=True, shuffle=True)

    with open("%s/S1/test_perceived_speech_wheretheressmoke.json"%(configs.DATA_JSON_PATH)) as json_file:
        test_loader  = IndexableDataset(data = json.load(json_file), batch_size=val_batch_size, shuffle=False)

    return train_loader, test_loader
