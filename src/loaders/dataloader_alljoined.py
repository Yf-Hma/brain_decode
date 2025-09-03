import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, IterableDataset
#from datasets import IterableDataset
import re
import random
from functools import partial


def pre_caption(caption):
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

    return caption
    


# class EEG2TextDataset(Dataset):
#     def __init__(self, dataset):
#         super(EEG2TextDataset, self).__init__()
#         self.dataset = dataset
#         self.annotations = np.load("exps/alljoined/COCO_73k_annots_curated.npy")
#         #self.annotations = np.load("exps/alljoined/COCO_73k_annots.npy")

#     def pre_caption(self, caption):
#         caption = re.sub(
#             r"([.!\"()*#:;~])",
#             " ",
#             caption.lower(),
#         )

#         caption = re.sub(
#             r"\s{2,}",
#             " ",
#             caption,
#         )
#         caption = caption.rstrip("\n")
#         caption = caption.strip(" ")

#         return caption
    
#     def __len__ (self):
#         return len (self.dataset)


#     def __getitem__(self, idx):
#         item = self.dataset.__getitem__(idx)
#         batch_signal = torch.tensor(item['EEG']).transpose(0, 1)
#         caption = self.annotations[item['73k_id']].tolist()
#         if type(caption) == list:
#             caption = [s for s in caption if len (s) > 0][0]

#         caption = self.pre_caption(caption)
#         return {"signal": batch_signal, "text_output": caption}


# class EEG2TextIterableDataset(IterableDataset):
#     def __init__(self, dataset, batch_size=1):
#         super(EEG2TextIterableDataset, self).__init__()
#         self.dataset = dataset

#         self.batch_size = batch_size

#         self.indices = self.dataset.__len__()
#         self.index_in_epoch = 0
#         self.annotations = np.load("exps/alljoined/COCO_73k_annots_curated.npy")

#     def pre_caption(self, caption):
#         caption = re.sub(
#             r"([.!\"()*#:;~])",
#             " ",
#             caption.lower(),
#         )

#         caption = re.sub(
#             r"\s{2,}",
#             " ",
#             caption,
#         )
#         caption = caption.rstrip("\n")
#         caption = caption.strip(" ")

#         return caption
    
#     def len (self):
#         return len (self.indices)


#     def __iter__(self):
#         self.index_in_epoch = 0
#         random.shuffle(self.indices)
#         return self

#     def __next__(self):
#         if self.index_in_epoch >= len(self.dataset):
#             raise StopIteration

#         batch_signals = []
#         batch_texts = []
#         num_items_in_batch = 0

#         while num_items_in_batch < self.batch_size and self.index_in_epoch < len(self.dataset):
#             idx = self.indices[self.index_in_epoch]
#             item = self._get_single_item(idx)
#             batch_signals.append(torch.tensor(item['EEG']).transpose(0, 1))

#             caption = self.annotations[item['73k_id']].tolist()

#             if type(caption) == list:
#                 caption = [s for s in caption if len (s) > 0][0]

#             caption = self.pre_caption(caption)

#             batch_texts.append(caption)
#             self.index_in_epoch += 1
#             num_items_in_batch += 1

#         if batch_signals:
#             batch_signals_tensor = torch.stack(batch_signals)
#             return {"signal": batch_signals_tensor, "text_output": batch_texts}
#         else:
#             raise StopIteration


    # def _get_single_item(self, idx):
    #     return self.dataset[idx]


# class MyCustomIterableDataset(IterableDataset):
#     def __init__(self, base_dataset, *args, **kwargs):
#         self.base_dataset = base_dataset
#         self.annotations = np.load("exps/alljoined/COCO_73k_annots_curated.npy")
#         super().__init__(ex_iterable=base_dataset, *args, **kwargs)

#     def __iter__(self):
#         # Get the iterator from the base dataset
#         iterator = iter(self.base_dataset)
#         # Iterate and yield modified items
#         for item in iterator:
#             batch_signal = torch.tensor(item['EEG']).transpose(0, 1)
#             caption = self.annotations[item['73k_id']].tolist()
#             if type(caption) == list:
#                 caption = [s for s in caption if len (s) > 0][0]

#             caption = pre_caption(caption)
#             yield {"signal": batch_signal, "text_output": caption}


annotations = np.load("exps/alljoined/COCO_73k_annots_curated.npy")
def update_items(item):
    batch_signal = torch.tensor(item['EEG']).transpose(0, 1)
    caption = annotations[item['73k_id']].tolist()
    if type(caption) == list:
        caption = [s for s in caption if len (s) > 0][0]

    caption = pre_caption(caption)
    return {"signal": batch_signal, "text_output": caption}


def to_tensor(item):
    item["signal"] = torch.stack(item["signal"])
    return item




class StreamingCodeDataset(IterableDataset):
    def __init__(self, DATA_PATH, batch_size, val_batch_size, split='train', buffer_size=10_000, world_size=1, rank=0):
        super().__init__()
        self.buffer_size = buffer_size
        self.world_size = world_size
        self.rank = rank

        # Set up dataset pipeline
        self.data_train = load_dataset(DATA_PATH, split=split, streaming=True)\
            .shard(num_shards=world_size, index=rank)\
            .shuffle(buffer_size=1000, seed=42)\
            .map(update_items)\
            .batch(batch_size)\
            .map(to_tensor)

    def __iter__(self):
        return iter(self.data_train)

def get_dada_loaders_all (DATA_PATH, batch_size, val_batch_size):

    train_set = StreamingCodeDataset(DATA_PATH, batch_size, val_batch_size)
    test_set = StreamingCodeDataset(DATA_PATH, batch_size, val_batch_size, split = "test")
    
    return train_set, test_set


if __name__ == '__main__':

    DATA_PATH = "/home/youssef.hmamouche/lustre/aim_neural-7he0p8agska/users/youssef.hmamouche/Alljoined/05_125/data"
    train_set, test_set = get_dada_loaders_all(DATA_PATH, 16, 16)

    for i, item in enumerate(train_set):
        print (item["signal"].shape)
        print (len (item["text_output"]))
        if i==10:
            break

