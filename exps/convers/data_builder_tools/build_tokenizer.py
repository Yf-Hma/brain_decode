
## importing the tokenizer and subword BPE trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit

from glob import glob
import os, sys

sys.path.insert(0, os.getcwd())
import configs.configs_convers as configs

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
    tokenizer.save("tools/tokenizer-convers.json")
    tokenizer = Tokenizer.from_file("tools/tokenizer-convers.json")
    return tokenizer

if __name__ == '__main__':

    text_files_right = glob ("%s/processed_data/interlocutor_text_data/**/*.txt"%configs.DATA_PATH, recursive=True)
    text_files_left = glob ("%s/processed_data/participant_text_data/**/*.txt"%configs.DATA_PATH, recursive=True)

    text_files =  text_files_left + text_files_right

    trained_tokenizer = train_tokenizer(text_files, "")

    input_string = "j'ai encore l'aubergine en forme de batman c'est la même c'était exactement la même c'est la même que tout à l'heure oui"#.replace ("'", " ")
    print (input_string)

    output = trained_tokenizer.encode(input_string, add_special_tokens=True)

    print(output)

    print (trained_tokenizer.decode(output.ids, skip_special_tokens = False))
