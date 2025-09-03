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
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit

sys.path.insert(0, os.getcwd())

from exps.perceived.utils.stimulus_utils import get_story_wordseqs, load_transcript

import configs.configs_perceived as configs


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
    tokenizer.save("tools/tokenizer-perceived.json")
    tokenizer = Tokenizer.from_file("tools/tokenizer-perceived.json")
    return tokenizer


def get_wordseqs_data(stories):
    word_seqs = get_story_wordseqs(stories, configs)

    new_word_sesq = []

    for story in stories:

        local_wrod_seqs = []
        j = 0

        new_time_index = word_seqs[story].tr_times[5 + configs.TRIM : -configs.TRIM] - (5 + configs.TRIM)

        for i in range (len (word_seqs[story].data)):
            if j >= len (new_time_index):
                break
            if word_seqs[story].data_times[i] <= new_time_index[j]:
                local_wrod_seqs.append (word_seqs[story].data[i])
            else:
                new_word_sesq.append (' '.join(local_wrod_seqs))
                local_wrod_seqs = []
                local_wrod_seqs.append (word_seqs[story].data[i])
                j += 1

    return new_word_sesq


def gather_all_text_train (args):

    # training stories
    stories = []
    with open(os.path.join(configs.DATA_TRAIN_PATH, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])


    # Textual data
    rtext_data = get_wordseqs_data (stories)

    return rtext_data


def gather_all_text_test (args):

    with open(os.path.join(configs.DATA_TEST_PATH, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)

    test_dict_data = []

    experiments = glob(os.path.join(configs.DATA_TEST_PATH, "test_stimulus/*"))
    experiments = [os.path.basename(x) for x in experiments]
    #print ([os.path.basename(x) for x in experiments])

    all_words = []
    for experiment in experiments:
        tasks = glob(os.path.join(configs.DATA_TEST_PATH, "test_stimulus", experiment, "*"))
        # print (os.path.join(args.data_path, configs.DATA_TEST_PATH, "test_stimulus", experiment, "*"))
        # exit ()

        tasks = [os.path.basename(x).split ('.')[0] for x in tasks]

        for task in tasks:
            transcript_data = load_transcript(experiment, task.split('_')[0], configs)
            ref_words, ref_times = transcript_data["words"], transcript_data["times"]
            all_words += ref_words.tolist()

    return all_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int,
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    words_text_test = gather_all_text_test (args)

    with open('words_test.txt', 'w') as f:
        for line in words_text_test:
            f.write(f"{line}\n")

    words_train = gather_all_text_train (args)


    with open('words_train.txt', 'w') as f:
        for line in words_train:
            f.write(f"{line}\n")


    trained_tokenizer = train_tokenizer(['words_train.txt'], "")
