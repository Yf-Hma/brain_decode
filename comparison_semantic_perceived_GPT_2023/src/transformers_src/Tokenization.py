import string
import json
import torch

def load_vocab_from_json(file_path):
    with open(file_path, 'r') as file:
        vocab = json.load(file)
    return vocab

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    text_without_pt = text.translate(translator)
    return text_without_pt


def get_filler_tokens_id(vocabulary):
    padding_token = "[PAD]"
    sos_token = "[SOS]"
    eos_token = "[EOS]"
    padding_token_id = vocabulary[padding_token]
    sos_token_id = vocabulary[sos_token]
    eos_token_id = vocabulary[eos_token]
    return padding_token_id, sos_token_id, eos_token_id


def add_filling_tokens_convert_to_tensor(token_id_list, sos_token_id, eos_token_id):
    token_ids_tensors = []
    for id_list in token_id_list:
        id_list = [sos_token_id] + id_list + [eos_token_id]
        id_tensor = torch.tensor(id_list)
        token_ids_tensors.append(id_tensor)
    return token_ids_tensors

