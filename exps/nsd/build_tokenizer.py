
## importing the tokenizer and subword BPE trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit

from glob import glob
import numpy as np
import argparse

import os, sys
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
# import src.configs_nsd as configs


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


def train_and_save_tokenizer(files, out_filename, alg='BPE'):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg)
    tokenizer.train(files, trainer) # training the tokenzier
    tokenizer.save(out_filename)
    tokenizer = Tokenizer.from_file(out_filename)
    return tokenizer

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
    args = parser.parse_args()

    all_captions = np.load('tools/COCO_73k_annots_curated.npy')

    out_filename = "tools/tokenizer_nsd.json"

    np.savetxt("coco.txt", all_captions,  fmt="%s")
    trained_tokenizer = train_and_save_tokenizer(["coco.txt"], out_filename, "")
    os.remove ("coco.txt")

    input_string = "test test test ---- Well that's quite good ----"#.replace ("'", " ")
    print (input_string)

    output = trained_tokenizer.encode(input_string, add_special_tokens=True)

    print(output)

    print (trained_tokenizer.decode(output.ids, skip_special_tokens = False))
