import torch
from src.Inference import inference1
from src.NLPLPIPSDataset import MyDataset
from src.Data_prepare import unzip_file, split_paths, read_split_append, read_statements, split_train_paths, split_val_paths, split_test_paths
from src.Tokenization import load_vocab_from_json, get_filler_tokens_id, add_filling_tokens_convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saving_file = 'SubmitionTest.txt'
model_path = 'M512/DeconvBipartiteTransformerConv.pt'
column_name = 'Dynamic Token ids'
vocab_path = 'speech_data/dynamic_statement.json'

fMRI_paths_train, fMRI_paths_val, fMRI_paths_test = split_paths('fMRI/fMRI_data/sub')

dialogues_paths_train = split_train_paths('speech_data/train/sub')
dialogues_paths_test = split_test_paths('speech_data/test/sub')
dialogues_paths = dialogues_paths_train + dialogues_paths_test


Tensor_list_train = read_split_append(fMRI_paths_train)
Tensor_list_test = read_split_append(fMRI_paths_test)
Tensor_list = Tensor_list_train + Tensor_list_test

Statements_train = read_statements(dialogues_paths, column_name)
Statements_test = read_statements(dialogues_paths_test, column_name)

############ Data Preparing and tokeization ##################

vocab = load_vocab_from_json(vocab_path)
flipped_vocab = {v: k for k, v in vocab.items()}
vocab_len = len(vocab)
padding_token_id, sos_token_id, eos_token_id = get_filler_tokens_id(vocab)


token_ids_tensors_train = add_filling_tokens_convert_to_tensor(Statements_train, sos_token_id, eos_token_id)
token_ids_tensors_test = add_filling_tokens_convert_to_tensor(Statements_test, sos_token_id, eos_token_id)

max_seq_length_train = max([tensor.size(-1) for tensor in token_ids_tensors_train])
max_seq_length_test = max([tensor.size(-1) for tensor in token_ids_tensors_test])

max_size = max(max_seq_length_train, max_seq_length_test)

############ Dataset innit #################

Dataset = MyDataset(Tensor_list, token_ids_tensors_train, sos_token_id, eos_token_id, max_size)
train_size = len(Dataset) - 250
Dataset_train = Dataset.slice(0, train_size)
Dataset_test = Dataset.slice(train_size, len(Dataset))

model = torch.load(model_path)

inference1(model, saving_file, vocab, Dataset_test, sos_token_id, eos_token_id, padding_token_id, 72, device)
