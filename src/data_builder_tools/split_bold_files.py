import os, shutil
from glob import glob
import pandas as pd
import numpy as np

import argparse

def split_tensor(tensor, sub_tensor_length, exclude_steps):
    #time_steps = tensor.size(0)
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
        #tensor = torch.from_numpy(tensor)
        tensor_list = split_tensor(tensor, 10, 3)
        for t in list(tensor_list):
            tensor_list_train.append(t)
    return tensor_list_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_data_path", default = "data/raw_data/convers_data/fMRI_data_200")
    args = parser.parse_args()


    if os.path.exists("data/processed_data/fMRI_data_split"):
        shutil.rmtree('data/processed_data/fMRI_data_split')

    os.makedirs('data/processed_data/fMRI_data_split')

    bold_files = glob ("%s/**/*.csv"%args.fmri_data_path, recursive=True)

    for filename in bold_files:

        data = read_bold_append ([filename])
        filename = filename.split (".csv")[0]
        filename_out = "data/processed_data/fMRI_data_split/" + filename.split ('/')[-2] + "_" + filename.split ('/')[-1]

        for i, data_split in enumerate(data):
            outfile = filename_out + "_split%d.npy"%(i+1)
            np.save(outfile, data_split)
