import argparse
import torch
import json
import numpy as np
import random
import os, sys
from transformers import set_seed

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

torch.cuda.empty_cache()
sys.path.insert(0, os.getcwd())

from src.models.models_nsd import BrainDEC_V0, BrainDEC_Clip
from src.load_nsd_data import get_loaders
import src.configs.nsd.configs as configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)


def main_single(ngpus_per_node, models_dict_type, data_path, src_fmri_features, args):
    print('Making datasets..')
    name = args.model_name + "_" + str (args.subj) + '_' + configs.LLM_name
    train_loader, val_loader, _, _ = get_loaders (data_path, args.subj,
                                        args.batch_size,
                                        args.val_batch_size,
                                        num_devices=1,
                                        rank = 0,
                                        world_size = 1
                                        )

    n_samples = 0
    for sample in train_loader:
        n_samples += 1

    print('Making model..')
    llm = models_dict_type[args.type](src_fmri_features, device = "cuda", load_in_4bit = args.load_in_4bit)
    optim = torch.optim.AdamW (llm.parameters(), lr = args.lr, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR (optim, max_lr=0.001, steps_per_epoch=n_samples, epochs=args.epochs_max)

    if args.retrain:
        args.starting_epoch, best_loss = load_from_checkpoint(llm, optim, lr_scheduler, args.saved_checkpoint, args.starting_epoch, 0)

    else:
        best_loss = 10000

    llm.train()
    print('Training..')
    train (llm,
           optim,
           lr_scheduler,
            name,
            configs.type,
            val_loader,
            [train_loader],
            configs.MODELS_TRAIN_PATH,
            0,
            src_fmri_features,
            args,
            n_samples = n_samples,
            epochs = args.epochs,
            save_epochs = args.save_epochs,
            starting_epoch = args.starting_epoch,
            lr = args.lr,
            best_loss = best_loss)

    print('..Done')



def main_worker(rank, ngpus_per_node, models_dict_type, data_path, src_fmri_features, args):

    torch.cuda.set_device(rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=rank)

    print('Making datasets..')

    assert os.path.exists (configs.LLM_PATH), "LLM_PATH, specified in config file, does not exist!"

    name = args.model_name + "_" + str (args.subj) + '_' + configs.LLM_name
    args.batch_size = int(args.batch_size / ngpus_per_node)

    train_loader, val_loader, _, _ = get_loaders (data_path, args.subj,
                                        args.batch_size,
                                        args.val_batch_size,
                                        num_devices=1,
                                        rank = rank,
                                        world_size=args.world_size
                                        )

    n_samples = 0
    for sample in train_loader:
        n_samples += 1
    print (n_samples)

    print('Making model..')
    llm = models_dict_type[args.type](src_fmri_features, device = "cuda:%d"%rank, load_in_4bit = args.load_in_4bit)
    optim = torch.optim.AdamW (llm.parameters(), lr = args.lr, betas=(0.9, 0.99))

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR (optim, max_lr=0.001, steps_per_epoch=n_samples, epochs=args.epochs)

    if args.retrain:
        args.starting_epoch, best_loss = load_from_checkpoint(llm, optim, lr_scheduler, args.saved_checkpoint, args.starting_epoch, rank)
    else:
        best_loss = 10000

    llm = llm.to(rank)
    llm = DDP(llm, device_ids=[rank], output_device=rank, find_unused_parameters = True)
    dist.barrier()

    llm.train()

    print('Training..')
    train (llm,
           optim,
           lr_scheduler,
            name,
            configs.type,
            val_loader,
            [train_loader],
            configs.MODELS_TRAIN_PATH,
            rank,
            src_fmri_features,
            args,
            n_samples = n_samples,
            epochs = args.epochs,
            save_epochs = args.save_epochs,
            starting_epoch = args.starting_epoch,
            lr = args.lr,
            best_loss = best_loss)

    print('..Done')
    dist.destroy_process_group()

def train (model, optim, lr_scheduler, model_name, type, val_loader, data_loaders,
           saving_path, rank, src_fmri_features, args, n_samples,
           epochs = 10, save_epochs = 1,
           starting_epoch = 1, lr = 0.0001, best_loss = 10000):

    model.train()
    for epoch in range(starting_epoch, epochs + 1):
        mean_loss = 0
        for id, sample in enumerate (data_loaders[-1]):
            padded = torch.zeros(sample[0].shape[0], sample[0].shape[1], src_fmri_features - sample[0].shape[2])
            sample[0] = torch.cat([sample[0],padded], dim = 2)

            loss = model (sample)
            mean_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()

        if args.distributed:
            dist.barrier()

        if epoch % save_epochs == 0:
            test_from_loader (val_loader, model, model_name, src_fmri_features, args, epoch)
            if rank == 0:
                print ('Loss: ', (mean_loss / n_samples))
                if (mean_loss / n_samples) < best_loss:
                    best_loss = mean_loss / n_samples
                    print(model_name + "_" + type, epoch, saving_path)
                if args.distributed:
                    save_checkpoint(model.module, optim, lr_scheduler, epoch, model_name, saving_path, best_loss)
                else:
                    save_checkpoint(model, optim, lr_scheduler, epoch, model_name, saving_path, best_loss)


def test_from_loader (data_loader, model, model_name, src_fmri_features, args, epoch):

    model.eval()
    if os.path.exists ("results/nsd/%s_%d.json"%(model_name, epoch)):
        os.remove ("results/nsd/%s_%d.json"%(model_name, epoch))

    results = {}
    sample_id = 0
    for sample in data_loader:

        src_fmri, _ = sample[0], sample[2].flatten().tolist()
        padded = torch.zeros(src_fmri.shape[0], src_fmri.shape[1], src_fmri_features - src_fmri.shape[2])
        src_fmri = torch.cat([src_fmri,padded], dim = 2)

        output_text = model.generate (src_fmri)

        output_text = [a.split('.')[0] + '.' for a in output_text]
        for a in  output_text:
            results[sample_id] = a
            sample_id += 1

    with open("results/nsd/%s_%d.json"%(model_name, epoch), 'w') as out_file:
        json.dump(results, out_file)

    model.train()



def test (data_path, models_dict_type, src_fmri_features, epoch = ""):
    model = models_dict_type[args.type](src_fmri_features, "cuda", load_in_4bit = args.load_in_4bit, inference_mode=True)
    model_name = args.saved_checkpoint.split('/')[-1].split('.')[0]
    checkpoint = torch.load(args.saved_checkpoint, map_location="cuda")


    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    _, data_loader, _, _ = get_loaders (data_path,
                                     args.subj,
                                     args.batch_size,
                                     args.val_batch_size,
                                     num_devices=1,
                                     rank=0,
                                     world_size=1)


    if os.path.exists ("results/nsd/%s.json"%model_name):
        os.remove ("results/nsd/%s.json"%model_name)

    results = {}
    sample_id = 0
    for sample in data_loader:
        src_fmri, _ = sample[0], sample[2]

        padded = torch.zeros(src_fmri.shape[0], src_fmri.shape[1], src_fmri_features - src_fmri.shape[2])
        src_fmri = torch.cat([src_fmri,padded], dim = 2)

        output_text = model.generate (src_fmri)

        output_text = [a.split('.')[0] + '.' for a in output_text]


        for a in  output_text:
            results[sample_id] = a
            sample_id += 1

    with open("results/nsd/%s.json"%model_name, 'w') as out_file:
        json.dump(results, out_file)



def save_checkpoint(model, optimizer, lr_scheduler, cur_epoch, model_name, saving_path, loss, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        'loss': loss.item(),
        "epoch": cur_epoch}

    save_to = "%s/%s_%s.pth"%(saving_path, model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, starting_epoch, rank=0):
    checkpoint = torch.load(checkpoint_path, map_location="cuda:%d"%rank)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)

    if "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "lr_scheduler" in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if "epoch" in checkpoint.keys():
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = starting_epoch +1

    if "loss" in checkpoint.keys():
        best_loss = checkpoint["loss"]
    else:
        best_loss = 10000

    return start_epoch, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("--val_batch_size", default = 128, type = int)
    parser.add_argument("--seed", default = 42, type=int)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.",
                        choices = ["MllmBrainToText_normal", "MllmBrainToText_deconv", "MllmBrainToText_clip_normal", "MllmBrainToText_clip_deconv"],
                        default = "MllmBrainToText_normal")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 5, type = int)
    parser.add_argument("--epochs", default = 240, type = int)
    parser.add_argument("--epochs_max", default = 240, type = int)
    parser.add_argument("--saved_checkpoint", "-s", type = str)
    parser.add_argument("--type", "-t", type = str, choices = ['normal', "clip"])
    parser.add_argument('--load_in_4bit', action='store_true', help = "to load the llm quantized in 4 bits for inference.")
    parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', default=False)

    parser.add_argument('--num_workers', type=int, default=1, help='')

    args = parser.parse_args()
    args.saving_path = configs.MODELS_TRAIN_PATH


    models_dict_type = {'normal':BrainDEC_V0, 'clip':BrainDEC_Clip}
    voxels_per_subj = {1: 15724, 2: 14280, 5: 13040, 7: 12685}
    src_fmri_features = 15724


    args.model_name = "BrainDEC_" + args.type
    data_path = configs.DATA_PATH

    if args.test:
        test (data_path, models_dict_type, src_fmri_features)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node

        if args.distributed:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, models_dict_type, data_path, src_fmri_features, args))
        else:
            ngpus_per_node = 1
            args.world_size = 1
            main_single(ngpus_per_node, models_dict_type, data_path, src_fmri_features, args)


    print (20 * '--', "Subject: ", args.subj, " ......Done")
