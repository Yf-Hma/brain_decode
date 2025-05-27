import json
import os
import sys
import re
import numpy
import torch
from torch.utils.data import Dataset, DataLoader


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import src.configs.perceived.configs as configs


class Tan2023CaptioningDataset(Dataset):
    def __init__(self, dataset):
        super(Tan2023CaptioningDataset, self).__init__()
        self.dataset = dataset
        self.prompt = ""
        self.max_words = 200
        self.device = "cuda"


    def __len__(self):
        return len(self.dataset)

    def pre_caption(self, caption):
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
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


    def __getitem__(self, idx):
        item = self.dataset[idx]

        bold_path = numpy.load (self.dataset[idx]["bold_signal"])
        bold_signal = torch.Tensor (bold_path)#.unsqueeze (0)
        caption = self.prompt + self.pre_caption(self.dataset[idx]["text-output"])

        if "chunk_number" in self.dataset[idx].keys():
            chunk_number = self.dataset[idx]["chunk_number"]
        else:
            chunk_number = -1


        return {"bold_signal": bold_signal, "text_input":"", "text_output":caption, "image": "", "chunk_number": chunk_number}

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict

def data_builder (subject, batch_size=32):
    with open("%s/%s/train.json"%(configs.PROCESSED_DATA_PATH,subject)) as json_file:
        train_dataset = Tan2023CaptioningDataset (json.load(json_file))

    #with open("data/%s/test.json"%subject) as json_file:
        #test_dataset = Tan2023CaptioningDataset (json.load(json_file))

    data_loaders = {}

    data_loaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    #data_loaders["test"]  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return data_loaders

def data_builder_v0 (batch_size=32):
    with open("data/train.json") as json_file:
        train_dataset = Tan2023CaptioningDataset (json.load(json_file))

    with open("data/test.json") as json_file:
        test_dataset = Tan2023CaptioningDataset (json.load(json_file))

    return train_dataset, test_dataset

def data_builder_from_file (filename, batch_size=32):
    with open(filename) as json_file:
        dataset = Tan2023CaptioningDataset (json.load(json_file))
    data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loaders


if __name__ == '__main__':
    data_loaders = data_builder ("S1", batch_size=32)

    print (len (data_loaders['train']))
    for element in data_loaders['train']:
        print (element['bold_signal'].shape)
        
