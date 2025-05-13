#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is an adapted file from utils.py: https://github.com/weihaox/UMBRAE"

import os
import argparse
# from pathlib import Path
import braceexpand

#import utils
import random
import numpy as np
import torch
import math
import json


import webdataset as wds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

# import warnings
# warnings.filterwarnings('ignore')

# tf32 data type is faster than standard float32

np.random.seed(42)

def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
    to_tuple=["voxels", "images", "trial"],
    subj=1,
    data_ratio=1.0,
    rank=0,
    world_size=1
):
    print("Getting dataloaders...")
    assert image_var == 'images'
    
    def my_split_by_node(urls):
        return urls
    
    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))

    train_url = train_url[rank::world_size]

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    

    num_samples = int(num_train * data_ratio)
    train_data = wds.WebDataset(train_url,
                                resampled=True,
                                cache_dir=cache_dir,
                                nodesplitter=my_split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .slice(num_samples)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)

    train_dl = DataLoader(train_data, batch_size=None, shuffle=False,  worker_init_fn=np.random.seed(42), pin_memory=True, num_workers=0)

    # validation (no shuffling, should be deterministic)  
    num_batches = math.floor(num_val / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    # print("\nnum_val", num_val)
    # print("val_num_batches", num_batches)
    # print("val_batch_size", val_batch_size)
    
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=True)

    val_dl = DataLoader(val_data, 
                        batch_size=None, 
                        shuffle=False)

    return train_dl, val_dl, num_train, num_val

def get_loaders (data_path, subj, batch_size, val_batch_size, num_devices, rank, world_size):

    train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"
    val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"

    meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
    num_train = 8559 + 300
    num_val = 982

    num_workers = num_devices
    print('\nprepare train and validation dataloaders...')
    train_dl, val_dl, num_train, num_val = get_dataloaders(
        batch_size, 'images',
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        num_train=num_train,
        num_val=num_val,
        val_batch_size=val_batch_size,
        cache_dir=data_path, 
        voxels_key='nsdgeneral.npy',
        to_tuple=["voxels", "images", "coco"],
        subj=subj,
        rank=rank, 
        world_size=world_size
    )

    return train_dl, val_dl, num_train, num_val


