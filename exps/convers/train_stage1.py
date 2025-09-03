import torch
import argparse, os
from tokenizers import Tokenizer
import sys
from transformers import set_seed
import torch.nn.functional as F
import json

sys.path.insert(0, os.getcwd())

import configs.configs_convers as configs
from src.loaders.dataloader_convers import get_loaders as convers_loader

from src.transformers_src.Transformer import *
from src.transformers_src.Train_Function import train_model

models_dict = {
'Transformer':Transformer,
'CNNTransformer':CNNTransformer,
'DuplexTransformerConv':DuplexTransformerConv,
'BipartiteTransformerConv':BipartiteTransformerConv,
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv,
}

def add_filling_tokens_convert_to_tensor(token_id_list, sos_token_id, eos_token_id, pad_token_id, max_size):
    token_ids_tensors = []
    for id_list in token_id_list:
        id_list = [sos_token_id] + id_list + [eos_token_id]
        token_id = [pad_token_id for _ in range (max_size)]
        for i in range (len (id_list)):
            if i < len (token_id):
                token_id[i] = id_list[i]
        if len (token_id) < len (id_list):
            token_id[-1] = eos_token_id
        token_ids_tensors.append(token_id)
    return torch.Tensor(token_ids_tensors).type(torch.int64)

def inference(model, saving_file, tokenizer, vocab_len, test_dataset, sos_token_id, eos_token_id, pad_token_id, max_seq_len, device):

    results = []
    for batch in test_dataset:
        # batch_end = min(batch_start + batch_size, train_num_samples)
        # batch = train_dataset[batch_start:batch_end]
        src, trg_sentences = batch["signal"], batch["text_output"]

        trg = []
        for a in trg_sentences:
            trg.append(tokenizer.encode(a, add_special_tokens=False).ids)

        trg = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
        src, trg = src.to(device), trg.to(device)

        output_ids, _, output = generate_sentence_ids(model, src.float(), sos_token_id,
                                                      eos_token_id, pad_token_id,
                                                      max_seq_len - 1, vocab_len, device)
        output = output.float()

        target, predicted = decode_sentences(output_ids, trg, tokenizer)

        results.append({"Generated": predicted, "Real": target})

    with open(saving_file, 'w') as out_file:
        json.dump(results, out_file)

def generate_sentence_ids(model, src, sos_token_id, eos_token_id, pad_token_id, max_length, vocab_len, device):
    model.eval()
    bs = src.size(0)
    # Preallocate memory
    sentences = torch.full((bs, max_length), pad_token_id, dtype=torch.float32, device=device)
    sentences[:, 0] = sos_token_id
    logits = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)
    prob = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)

    # Encoder once
    e_outputs, src_mask = model.encoder(src.float())
    with torch.no_grad():
        for t in range(1, max_length): #max_length
            current_tokens = sentences[:, :t]
            current_tokens_padded = torch.nn.functional.pad(current_tokens, (0, max_length - t), value=pad_token_id)
            d_output = model.decoder(current_tokens_padded.float(), e_outputs, src_mask)
            output = model.out(d_output)

            next_logit = output[:, t, :]
            logits[:, t, :] = next_logit

            softmax_output = F.softmax(output, dim=2)

            next_prob = softmax_output[:, t, :]
            prob[:, t, :] = next_prob

            next_tokens = next_prob.argmax(dim=-1)
            
            sentences[:, t] = int (next_tokens)

    return sentences.to(device), prob.to(device), logits.to(device)

def decode_sentences(output, trg_des, tokenizer):
    trg_des = trg_des.flatten().tolist()
    output_words = tokenizer.decode(output[0].type(torch.int64).tolist(), skip_special_tokens = True)#.split (' ')
    desc_words = tokenizer.decode(trg_des, skip_special_tokens = True)#.split (' ')
    #output_sentence = ' '.join(output_words)
    #desc_sentence = ' '.join(desc_words)
    return desc_words, output_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys(), default = "DeconvBipartiteTransformerConv")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--seed", default = 1, type=int)
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--type", "-t", type = str, default = 'spoken', choices = ['spoken', 'perceived'])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('results/convers'):
        os.mkdir('results/convers')

    batch_size = args.batch_size
    wandb_log = False
    epochs = args.epochs
    lr = args.lr

    src_fmri_features = configs.src_fmri_features
    time_steps = configs.time_steps
    max_size = configs.max_size
    d_model = configs.d_model
    heads = configs.heads
    d_ff = configs.d_ff
    N = configs.N

    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    tokenizer = Tokenizer.from_file("./tools/tokenizer-convers.json")
    vocab_len = tokenizer.get_vocab_size()

    # TODO: increase batch size in testing
    if args.test:
        batch_size = 1

    ################ Datasets ##############
    train_set_convers, test_set_convers = convers_loader(batch_size = batch_size, val_batch_size = batch_size)

    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, 
                        src_fmri_features, 
                        max_size, 
                        vocab_len, 
                        d_model, 
                        d_ff, 
                        N, 
                        heads, 
                        device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ################ Model Training/Testing ##############
    seed = args.seed
    set_seed(seed)

    name = args.model_name  + '_' + str(src_fmri_features)
    out_name = os.path.join (configs.MODELS_TRAIN_PATH, name)

    if args.test:
        model.load_state_dict(torch.load('%s.pt'%(out_name), weights_only=True))

        if not os.path.exists('results/convers/%s'%(args.model_name)):
            os.mkdir('results/convers/%s'%(args.model_name))

        saving_file = 'results/convers/%s/%s.json'%(args.model_name, name)
        inference(model, saving_file, tokenizer, vocab_len, test_set_convers, sos_token_id, eos_token_id, pad_token_id, max_size, device)
    else:
        if args.retrain:
            assert os.path.exists('%s.pt'%(out_name)), "Checkpoint does not exists."
            checkpoint = torch.load('%s.pt'%(out_name), map_location=torch.device(device))
            model.load_state_dict(checkpoint)
        model = train_model(out_name, model, train_set_convers, batch_size, optim, epochs, lr, N, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device, wandb_log)
