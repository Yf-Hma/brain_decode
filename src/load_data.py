import json
import numpy
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# from torchvision import transforms
from CLIP import clip

import src.configs as configs

class BoldCaptioningDataset(Dataset):
    def __init__(self, dataset):
        super(BoldCaptioningDataset, self).__init__()
        self.dataset = dataset
        self.prompt = ""
        self.max_words = 500
        _, self.preprocess = clip.load("ViT-B/32", device="cuda")
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
        query = self.pre_caption(self.dataset[idx]["text-input"])

        try:
            image = Image.open(self.dataset[idx]["image"])
            image = image.resize((224,224))
            images_input = self.preprocess(image)
        except:
            images_input = torch.Tensor([0.])

        return {"bold_signal": bold_signal, "text_input":query, "text_output":caption, "image": images_input}

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

def data_builder (batch_size=32):
    with open("%s/train.json"%configs.JSON_DATA_PATH_OUT) as json_file:
        train_dataset = BoldCaptioningDataset (json.load(json_file))

    with open("%s/test.json"%configs.JSON_DATA_PATH_OUT) as json_file:
        test_dataset = BoldCaptioningDataset (json.load(json_file))

    data_loaders = {}

    data_loaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_loaders["test"]  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return data_loaders

def data_builder_v0 (batch_size=32):
    with open("%s/train.json"%configs.JSON_DATA_PATH_OUT) as json_file:
        train_dataset = BoldCaptioningDataset (json.load(json_file))

    with open("%s/test.json"%configs.JSON_DATA_PATH_OUT) as json_file:
        test_dataset = BoldCaptioningDataset (json.load(json_file))

    return train_dataset, test_dataset
