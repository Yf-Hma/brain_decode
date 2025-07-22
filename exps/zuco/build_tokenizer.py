import os
import json
import argparse
import sys
from glob import glob

## importing the tokenizer and subword BPE trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import CharDelimiterSplit


sys.path.insert(0, os.getcwd())

import src.configs.zuco.configs as configs
from exps.zuco.load_zuco_data import get_loaders


unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<PAD>", "<SOS>", "<EOS>", "<MASK>", "<UNK>"]  # special tokens

def prepare_tokenizer_trainer(alg):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token = unk_token))
        trainer = BpeTrainer(special_tokens = spl_tokens)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token= unk_token, special_tokens = spl_tokens)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
        trainer = WordPieceTrainer(special_tokens = spl_tokens)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token = unk_token))
        trainer = WordLevelTrainer(special_tokens = spl_tokens)

    tokenizer.pre_tokenizer = CharDelimiterSplit(" ")
    return tokenizer, trainer


def train_tokenizer(files, alg='BPE'):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg)
    tokenizer.train(files, trainer) # training the tokenzier
    tokenizer.save("tools/tokenizer-zuco.json")
    tokenizer = Tokenizer.from_file("tools/tokenizer-zuco.json")
    return tokenizer





if __name__ == "__main__":

    ################## Load Data ####################################
    train_set, test_set = get_loaders (configs.PROCESSED_DATA_PATH, 64, 64)
    all_text = []

    ################## Gather text from Train Data ###################
    for item in train_set:
        print (item["signal"].shape)
        print (item["text_output"])
        all_text += item["text_output"]

    with open('text_zuco.txt', 'w') as f:
        for line in all_text:
            f.write(f"{line}\n")

    # with open('words_train.txt', 'w') as f:
    #     for line in words_train:
    #         f.write(f"{line}\n")

    trained_tokenizer = train_tokenizer(['words_train.txt'], "")

    os.remove ('text_zuco.txt')
