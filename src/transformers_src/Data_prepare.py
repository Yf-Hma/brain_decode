import os
import zipfile
import pandas as pd
import torch
import numpy as np


def unzip_file(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        os.makedirs(extract_dir)
        zip_ref.extractall(extract_dir)


'************************** train test val split **************************'


def split_paths(file_path_format):
    paths_train = []
    for i in range(1, 10):
        for root, dirs, files in os.walk(f'{file_path_format}-0{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_train.append(filepath)
    for i in range(10, 24):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_train.append(filepath)
    paths_val = []
    for i in range(20, 23):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_val.append(filepath)
    paths_test = []
    for i in range(24, 26):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_test.append(filepath)
    return paths_train, paths_val, paths_test

def split_train_paths(file_path_format):
    paths_train = []
    for i in range(1, 10):
        for root, dirs, files in os.walk(f'{file_path_format}-0{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_train.append(filepath)
    for i in range(10, 24):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_train.append(filepath)
    return paths_train

def split_val_paths(file_path_format):
    paths_val = []
    for i in range(20, 23):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_val.append(filepath)
    return paths_val

def split_test_paths(file_path_format):
    paths_test = []
    for i in range(24, 26):
        for root, dirs, files in os.walk(f'{file_path_format}-{i}'):
            for name in files:
                filepath = root + os.sep + name
                paths_test.append(filepath)
    return paths_test

'************************** Splitting the fMRI tensors to 14 time steps each **************************'


# def split_tensor(tensor, window_size, stride):
#     num_sub_tensors = (tensor.size(0) - window_size) // stride + 1
#     sub_tensors = []
#     for i in range(1, num_sub_tensors):
#         start_index = i * stride
#         end_index = start_index + window_size
#         sub_tensor = tensor[start_index:end_index]
#         sub_tensors.append(sub_tensor)
#     return sub_tensors

def split_tensor(tensor, sub_tensor_length, exclude_steps):
    time_steps = tensor.size(0)
    removed_tensor = tensor[exclude_steps:]
    num_tensors = (time_steps - exclude_steps) // sub_tensor_length
    divided_tensors = []
    
    for i in range(num_tensors):
        start = i * sub_tensor_length
        end = start + sub_tensor_length
        sub_tensor = removed_tensor[start:end]
        divided_tensors.append(sub_tensor)
    
    return divided_tensors

def read_split_append(fmri_paths):
    tensor_list_train = []
    for fmri_path in fmri_paths:
        df = pd.read_csv(fmri_path, delimiter=';')
        tensor = np.array(df.iloc[0:, 1:-1].values)
        tensor = torch.from_numpy(tensor)
        tensor_list = split_tensor(tensor, 10, 3)
        for t in list(tensor_list):
            tensor_list_train.append(t)
    return tensor_list_train


def read_statements(dialogues_paths, Column):
    statements = []
    for Dialogue_path in dialogues_paths:
        df = pd.read_csv(Dialogue_path, dtype={'list_column': object})
        df[Column] = df[Column].apply(eval)
        statement_list = df[Column].to_list()
        for s in statement_list:
            statements.append(s)
    return statements

