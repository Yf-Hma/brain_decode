import json
import sys
import os
import numpy
import re
import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torch.utils.data import IterableDataset
import random

sys.path.insert(0, os.getcwd())

import clip
import configs.configs_convers as configs


_, preprocessor = clip.load("ViT-B/32", device="cuda")


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
    # if len (bold_signal.shape) == 3:
    #     padded = torch.zeros(bold_signal.shape[0], bold_signal.shape[1], configs.src_fmri_features_max - bold_signal.shape[2])
    #     bold_signal = torch.cat([bold_signal,padded], dim = 2)
    # elif len(bold_signal.shape) == 2:
    #     padded = torch.zeros(bold_signal.shape[0], configs.src_fmri_features_max - bold_signal.shape[1])
    #     bold_signal = torch.cat([bold_signal,padded], dim = 1)

    return bold_signal


def make_sample(item, version):

    if version == "V2":
        image = Image.open(item["image"]).resize((224,224))
        images_input = preprocessor(image)

        try:
            image = Image.open(item["image"])
            image = image.resize((224,224))
            images_input = preprocessor(image)
        except:
            images_input = torch.empty((1), dtype=torch.int32)
    else:
        images_input = torch.empty((1), dtype=torch.int32)

    sample = {
        "text_output": item["text-output"],
        "text_input": item["text-input"],
        "signal": read_signal(item["bold_signal"]),
        "image": images_input,
    }
    return sample



class IndexableDataset(IterableDataset):
    def __init__(self, data, world_size=1, rank=0, transform=None,
                 shuffle=False, batch_size=None, seed=None, drop_last = False, version="V0"):

        self.data = data
        self.world_size = world_size
        self.rank = rank
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_mode = True
        self.seed = seed
        self.drop_last = drop_last
        self.version = version

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

        batch = {"text_output":[], "signal": [], "text_input": [], "image": []}

        for item in data:
            sample = make_sample(item, self.version)

            batch["text_output"].append(sample["text_output"])
            batch["signal"].append(sample["signal"])
            batch["text_input"].append(sample["text_input"])
            batch["image"].append(sample["image"])

            #print (len(batch["text_output"]))
            if len(batch["text_output"]) == batch_size:
                batch["signal"] = torch.stack(batch["signal"])
                batch["image"] = torch.stack(batch["image"])
                yield batch

                batch = {"text_output":[], "signal": [], "text_input": [], "image": []}

        if len (batch["text_output"]) > 0:
            if  len (batch["text_output"]) == self.batch_size:
                batch["signal"] = torch.stack(batch["signal"])
                batch["image"] = torch.stack(batch["image"])
                yield batch

            if  self.drop_last==False:
                batch["signal"] = torch.stack(batch["signal"])
                batch["image"] = torch.stack(batch["image"])
                yield batch


####################### Get Loaders #######################
def get_loaders (DATA_PATH = configs.DATA_PATH, batch_size=32, val_batch_size=32, version="V0"):

    assert version in ["V0", "V1", "V2"], 'version must be in ["V0", "V1", "V2"]'
    with open("%s/train.json"%DATA_PATH) as json_file:
        train_loader = IndexableDataset(data = json.load(json_file), batch_size=batch_size, shuffle=True, drop_last=True, version=version)

    with open("%s/test.json"%DATA_PATH) as json_file:
        test_loader = IndexableDataset(data = json.load(json_file), batch_size=val_batch_size, shuffle=False, version=version)

    return train_loader, test_loader

##############################################
if __name__ == '__main__':

    train_set, test_set = get_loaders(configs.DATA_PATH, 16, 16, version="V0")

    for i, item in enumerate(train_set):
        # print (item["signal"].shape)
        # print (len (item["text_output"]))
        if i==1:
            break
