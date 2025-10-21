import torch
import torch.nn as nn
import random
import os, sys, argparse
from tokenizers import Tokenizer

sys.path.insert(0, os.getcwd())

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv
from src.loaders.dataloader_nsd import get_loaders as nsd_loader
import configs.configs_nsd as configs
import random
import numpy as np



def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')


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


def train_model(out_path, model, train_datasets, optimizer, num_epochs, lr, N, sos_token_id, eos_token_id, pad_token_id, max_seq_len, tokenizer, device):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0.0
        total_samples_training = 0

        random.shuffle (train_datasets)
        for train_dataset in train_datasets:
            
            for batch in train_dataset:
                src, trg_sentences = batch["signal"], batch["text_output"]

                padded = torch.zeros(src.shape[0], src.shape[1], 17910 - src.shape[2])
                src = torch.cat([src,padded], dim = 2)

                trg = []
                for a in trg_sentences:
                    if len (a) > 0:
                        trg.append(tokenizer.encode(a, add_special_tokens=False).ids)
                    else:
                        trg.append(tokenizer.encode("", add_special_tokens=False).ids)

                input_decoder = [a[:-1] for a in trg]
                input_decoder = add_filling_tokens_convert_to_tensor(input_decoder, sos_token_id, eos_token_id, pad_token_id, max_seq_len - 1)
                label_decoder = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
                label_decoder = label_decoder[:,1:]

                src, input_decoder = src.to(device), input_decoder.to(device)
                label_decoder = label_decoder.to(device)
                optimizer.zero_grad()  # Clear gradients

                output, softmax_output = model(src.float(),input_decoder.float())
                loss = criterion(output.reshape(-1, output.size(-1)), label_decoder.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss_train += loss.item() * label_decoder.size(0)  # Accumulate loss
                total_samples_training += label_decoder.size(0)  # Accumulate number of samples

        epoch_loss = total_loss_train / total_samples_training
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), '%s_%d.pt'%(out_path, epoch))
            torch.save(model.state_dict(), '%s.pt'%(out_path))
            if epoch > 10:
                os.system ("rm %s_%d.pt"%(out_path, epoch - 10))

    print("Training completed!")
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--seed", default = 42, type=int)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--val_batch_size", default = 128, type = int)
    parser.add_argument("--epochs", default = 40, type = int)
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    seed_everything (args.seed)
    
    
    ################ Parameters Init ##############
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    
    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    src_fmri_features = configs.src_fmri_features
    time_steps = configs.time_steps
    data_path = configs.DATA_PATH
    max_size = configs.max_size
    d_model = configs.d_model
    heads = configs.heads
    d_ff = configs.d_ff
    N = configs.N

    tokenizer = Tokenizer.from_file("tools/tokenizer_nsd.json")
    vocab_len = tokenizer.get_vocab_size()
    coco_captions_file = np.load('tools/COCO_73k_annots_curated.npy')


    ################ Datasets ##############
    train_loader, val_loader, _, _ = nsd_loader (data_path, args.subj,
                                        args.batch_size,
                                        args.val_batch_size,
                                        num_devices=1,
                                        rank = 0,
                                        world_size = 1,
                                        coco_captions_file = coco_captions_file
                                        )
    
    train_loader, val_loader, _, _ = nsd_loader (data_path, 1, 
                                        args.batch_size, 
                                        args.val_batch_size, 
                                        num_devices=1,
                                        rank = 0, 
                                        world_size = 1,
                                        coco_captions_file = coco_captions_file
                                        )

    train_loader_2, _, _, _ = nsd_loader (data_path, 2, 
                                        args.batch_size, 
                                        args.val_batch_size, 
                                        num_devices=1,
                                        rank = 0, 
                                        world_size = 1,
                                        coco_captions_file = coco_captions_file
                                        )


    train_loader_5, _, _, _ = nsd_loader (data_path, 5, 
                                        args.batch_size, 
                                        args.val_batch_size, 
                                        num_devices=1,
                                        rank = 0, 
                                        world_size = 1,
                                        coco_captions_file = coco_captions_file
                                        )

    train_loader_7, _, _, _ = nsd_loader (data_path, 7, 
                                        args.batch_size, 
                                        args.val_batch_size, 
                                        num_devices=1,
                                        rank = 0, 
                                        world_size = 1,
                                        coco_captions_file = coco_captions_file
                                        )


    ################ Model Init ##############
    model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ################ Model Training ##############
    name = args.model_name  + '_' + str(src_fmri_features) + '_' + configs.type
    out_name = os.path.join (configs.MODELS_TRAIN_PATH, name)


    if args.retrain:
        model = torch.load('%s.pt'%(out_name), map_location=torch.device(device))
        
    model = train_model(out_name, model, 
                        [train_loader, train_loader_2, train_loader_5, train_loader_7], 
                        optim, epochs, 
                        lr, 
                        N, 
                        sos_token_id, 
                        eos_token_id, 
                        pad_token_id,
                        max_size, 
                        tokenizer, 
                        device)


