from glob import glob
import pandas as pd
import json, os, sys
import numpy as np
import random
import shutil

sys.path.insert(0, os.getcwd())
import src.configs.convers.configs as configs


def load_vocab_from_json(file_path):
    with open(file_path, 'r') as file:
        vocab = json.load(file)
    return vocab


def detokenize(token_ids, vocab_dict):
    word_list = []
    for tid in token_ids:
        word = vocab_dict[int(tid)]
        word_list.append(word)
    sentence = ' '.join(word_list)
    return sentence

def read_statements(dialogues_paths, Column):
    statements = []
    for Dialogue_path in dialogues_paths:
        df = pd.read_csv(Dialogue_path, dtype={'list_column': object})
        df[Column] = df[Column].apply(eval)
        statement_list = df[Column].to_list()
        for s in statement_list:
            statements.append(s)
    return statements


def text_files_to_dic (text_files):
    out_dict = []

    for filename_left in text_files:
        image_path = "%s/raw_data/images/"%configs.DATA_PATH + filename_left.split ('-')[-1].split ('.')[0] + ".jpg"
        filename_right = filename_left.replace ("participant_text_data", "interlocutor_text_data")

        with open(filename_right) as file:
            lines_right = [line.rstrip() for line in file]

        with open(filename_left) as file:
            lines_left = [line.rstrip() for line in file]

        while len (lines_right) < 5:
            lines_right.append (" ")

        while len (lines_left) < 5:
            lines_left.append (" ")

        for i in range (5):
            associated_bold_file = "%s/processed_data/fMRI_data_split/"%configs.DATA_PATH + filename_right.split ('/')[-1].split (".")[0] + "_split%d.npy"%(i+1)
            out_dict.append ({"image": image_path, "bold_signal": associated_bold_file, "text-output": lines_left[i], "text-input": lines_right[i]})
    return out_dict

if __name__ == "__main__":

    random.seed(42)
    text_files = sorted (glob ("%s/processed_data/participant_text_data/**/*.txt"%configs.DATA_PATH, recursive=True))

    random.shuffle(text_files)

    text_files_train = text_files[:-50]
    text_files_test = text_files[-50:]

    train_list = text_files_to_dic (text_files_train)
    test_list = text_files_to_dic (text_files_test)

    with open('%s/train.json'%configs.DATA_PATH, 'w') as output_file:
        json.dump(train_list, output_file)

    with open('%s/test.json'%configs.DATA_PATH, 'w') as output_file:
        json.dump(test_list, output_file)
