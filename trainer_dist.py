import argparse
import os
import torch
from src.load_data import data_builder_v0, data_builder
from src.models import MllmBrainToTextV0, MllmBrainToText, MllmBrainToTextV2

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

import src.config as configs


def main_worker(gpu, ngpus_per_node, models_dict, args):

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu    

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('Making datasets..')
    torch.cuda.set_device(args.gpu)    
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    dataset_train, dataset_test = data_builder_v0(args.batch_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              sampler=train_sampler)

    print('Making model..')
    if args.retrain:
        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        llm = models_dict[args.model_name]()
        llm = load_from_checkpoint(llm, args.saved_checkpoint)
    else:
        llm = models_dict[args.model_name]()

    llm.cuda(args.gpu)
    llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[args.gpu], find_unused_parameters = True)
    num_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)    

    train (llm, 
            args.model_name,
            configs.type,
            train_loader,
            configs.MODELS_TRAIN_DIR,
            args.rank,
            epochs = args.epochs,
            save_epochs = args.save_epochs,
            starting_epoch = args.starting_epoch,
            )
    
    # dist.barrier()
    # if args.rank == 0:
    #     save_checkpoint(llm.module, args.model_name, "0")
    #     test (llm.module.to(0), test_loader, args.model_name)
          
    # if args.rank == 0:
    #     test (llm.to(0), test_loader, args.model_name)
    


def save_checkpoint(model, model_name, cur_epoch, saving_path, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    os.system ("rm %s/%s*"%(saving_path, model_name))
    save_to = "%s/%s_%s.pth"%(saving_path, model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)

def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, type, data_loader, saving_path, rank, epochs = 100, save_epochs = 10, starting_epoch = 1):

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
     
    best_loss = 100000

    for epoch in range(starting_epoch, epochs + 1):
        if rank == 0:
            print ('-------- Epoch: ', epoch)
        mean_loss = 0
        # if epoch > (epochs - 20):
        #     save_epochs = 2

        for sample in data_loader:
            loss = model(sample)
            mean_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    
        dist.barrier()

        if epoch % save_epochs == 0:    
            if rank == 0:
                if (mean_loss / len (data_loader)) < best_loss:
                    best_loss = mean_loss
                save_checkpoint(model, model_name + "_" + type, epoch, saving_path)
            

def test (model, data_loader, model_name):
    model.eval()
    f = open("results/%s.txt"%model_name, "w")

    for sample in data_loader:
        output_text = model.generate (sample)
        for predicted, target in zip (output_text, sample["text_output"]):
            f.write("The predicted Conversation :")
            f.write(predicted + "\n")
            f.write("The target Conversation :")
            f.write(target + "\n")

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("-seed", default = 3)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["MllmBrainToTextV0", "MllmBrainToTextV1", "MllmBrainToTextV2"], default = "MllmBrainToTextV0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 5, type = int)
    parser.add_argument("--epochs", default = 300, type = int)
    parser.add_argument("--saved_checkpoint", "-s", type = str)
    parser.add_argument("--saving_path", default = "trained_models")

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    args = parser.parse_args()

    models_dict = {
    'MllmBrainToTextV0':MllmBrainToTextV0,
    'MllmBrainToTextV1':MllmBrainToText,
    'MllmBrainToTextV2':MllmBrainToTextV2,
    }

    torch.manual_seed(args.seed)

    if args.test:
        args.batch_size = 32
        data_loader = data_builder(args.batch_size)
        llm = models_dict[args.model_name]()
        llm = load_from_checkpoint(llm, args.saved_checkpoint).to(0)
        test (llm, data_loader["test"], args.model_name)

    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, models_dict, args))
