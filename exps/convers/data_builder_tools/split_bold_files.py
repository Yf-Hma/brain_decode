import os, shutil, sys
from glob import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())

import src.configs.convers.configs as configs

def split_tensor(tensor, sub_tensor_length, exclude_steps):
    time_steps = tensor.shape[0]
    removed_tensor = tensor[exclude_steps:]
    num_tensors = (time_steps - exclude_steps) // sub_tensor_length
    divided_tensors = []

    for i in range(num_tensors):
        start = i * sub_tensor_length
        end = start + sub_tensor_length
        sub_tensor = removed_tensor[start:end]
        divided_tensors.append(sub_tensor)

    return divided_tensors

def read_bold_append(fmri_paths):
    tensor_list_train = []
    for fmri_path in fmri_paths:
        df = pd.read_csv(fmri_path, delimiter=',')
        tensor = np.array(df.iloc[0:, 1:].values)
        tensor_list = split_tensor(tensor, 10, 3)
        for t in list(tensor_list):
            tensor_list_train.append(t)
    return tensor_list_train


if __name__ == "__main__":

    
    if os.path.exists("%s/processed_data/fMRI_data_split"%configs.DATA_PATH):
        shutil.rmtree('%s/processed_data/fMRI_data_split'%configs.DATA_PATH)

    os.makedirs('%s/processed_data/fMRI_data_split'%configs.DATA_PATH)
    
    bold_files = glob ("%s/fMRI_data_%s/**/*.csv"%(configs.PROCESSED_FMRI_DATA_PATH, str(configs.src_fmri_features)), recursive=True)


    for filename in bold_files:

        data = read_bold_append ([filename])
        filename = filename.split (".csv")[0]
        filename_out = os.path.join ("%s/processed_data/fMRI_data_split/"%configs.DATA_PATH, filename.split ('/')[-2] + "_" + filename.split ('/')[-1])

        for i, data_split in enumerate(data):
            outfile = filename_out + "_split%d.npy"%(i+1)
            np.save(outfile, data_split)
