import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import sys

sys.path.insert(0, os.getcwd())

import src.configs.zuco.configs as configs

parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
args = vars(parser.parse_args())


"""config"""
version = 'v1' # 'old'
# version = 'v2' # 'new'

task_name = args['task_name']
# task_name = 'task1-SR'
# task_name = 'task2-NR'
# task_name = 'task3-TSR'


print('##############################')
print(f'start processing ZuCo {task_name}...')

data_path = configs.DATA_PATH

outdir = f"{data_path}/processed"

if version == 'v1':
    # old version 
    input_mat_files_dir = f'{data_path}/{task_name}/Matlab_files' 
elif version == 'v2':
    # new version, mat73 
    input_mat_files_dir = f'{data_path}/{task_name}/Matlab_files' 

output_dir = f'%s/{task_name}/pickle'%outdir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    if version == 'v1':
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
    elif version == 'v2':
        matdata = h5py.File(mat_file,'r')
        print(matdata)

    for e,sent in enumerate(matdata):
        word_data = sent.word
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content':sent.content}
            sent_obj['sentence_level_EEG'] = {'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2, 'mean_a1':sent.mean_a1, 'mean_a2':sent.mean_a2, 'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2, 'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2}

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {'answer_mean_t1':sent.answer_mean_t1, 'answer_mean_t2':sent.answer_mean_t2, 'answer_mean_a1':sent.answer_mean_a1, 'answer_mean_a2':sent.answer_mean_a2, 'answer_mean_b1':sent.answer_mean_b1, 'answer_mean_b2':sent.answer_mean_b2, 'answer_mean_g1':sent.answer_mean_g1, 'answer_mean_g2':sent.answer_mean_g2}
            
            # word level:
            sent_obj['word'] = []
            
            word_tokens_has_fixation = [] 
            word_tokens_with_mask = []
            word_tokens_all = []


            for word in word_data:
                if isinstance(word.nFixations, np.ndarray) or isinstance(word.nFixations, list):
                    if len (word.nFixations) == 0:
                        word.nFixations = 0

                word_obj = {'content':word.content}
                word_tokens_all.append(word.content)
                # TODO: add more version of word level eeg: GD, SFD, GPT
                word_obj['nFixations'] = word.nFixations
                if word.nFixations > 0:    
                    word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 'FFD_a1':word.FFD_a1, 'FFD_a2':word.FFD_a2, 'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2}}
                    word_obj['word_level_EEG']['TRT'] = {'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 'TRT_a1':word.TRT_a1, 'TRT_a2':word.TRT_a2, 'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2}
                    word_obj['word_level_EEG']['GD'] = {'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 'GD_a1':word.GD_a1, 'GD_a2':word.GD_a2, 'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2}
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                    #print(word.rawEEG)
                    if type(word.rawEEG) == np.ndarray:
                        """print(e)
                        print(f"content: {word.content}")
                        print(f"shape: {word.rawEEG.shape}")
                        print(f"size: {word.rawEEG.size}")
                        print(f"ndim: {word.rawEEG.ndim}")
                        print("nan:",np.isnan(word.rawEEG[0]).any())"""
                        if not np.isnan(word.rawEEG[0]).any():
                            if word.rawEEG[0].ndim == 2:

                                word_obj['rawEEG']=[np.float32(arr.transpose()) for arr in word.rawEEG if type(arr)==np.ndarray]
                            elif word.rawEEG[0].ndim == 1:

                                word_obj['rawEEG']=[word.rawEEG.transpose()]
                        else:
                            word_obj['rawEEG']=[]
                    else:
                        word_obj['rawEEG']=[]
                    """ print("len:",len(word_obj['rawEEG']))
                    print("\n")"""

                    #print(word_obj['rawEEG'])
                else:
                    word_tokens_with_mask.append('[MASK]')
                    # if a word has no fixation, use sentence level feature
                    # word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':sent.mean_t1, 'FFD_t2':sent.mean_t2, 'FFD_a1':sent.mean_a1, 'FFD_a2':sent.mean_a2, 'FFD_b1':sent.mean_b1, 'FFD_b2':sent.mean_b2, 'FFD_g1':sent.mean_g1, 'FFD_g2':sent.mean_g2}}
                    # word_obj['word_level_EEG']['TRT'] = {'TRT_t1':sent.mean_t1, 'TRT_t2':sent.mean_t2, 'TRT_a1':sent.mean_a1, 'TRT_a2':sent.mean_a2, 'TRT_b1':sent.mean_b1, 'TRT_b2':sent.mean_b2, 'TRT_g1':sent.mean_g1, 'TRT_g2':sent.mean_g2}
                    
                    # NOTE:if a word has no fixation, simply skip it
                    continue
            
            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            
            dataset_dict[subject_name].append(sent_obj)

        else:
            print(f'missing sent: subj:{subject_name} content:{sent.content}, return None')
            dataset_dict[subject_name].append(None)

            continue
    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])

"""output"""
output_name = f'{task_name}-dataset_wRaw.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))


"""sanity check"""
# check dataset
with open(os.path.join(output_dir,output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

if version == 'v1':
    print('num of sent:', len(whole_dataset['ZAB']))
    print()