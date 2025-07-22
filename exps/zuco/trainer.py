import torch
import os, json
import torch.distributed as dist

############# Training on a single GPU #############
def train_single(model, out_name, train_loader, test_loader, args, MODELS_TRAIN_DIR):

    print('Making datasets..')

    n_samples = 0
    for sample in train_loader:
        n_samples += 1

    print ("Number of samples: ", n_samples)

    optim = torch.optim.AdamW (model.parameters(), lr = args.lr, betas=(0.9, 0.99))
    lr_scheduler = None

    if args.retrain:
        args.starting_epoch, best_loss = load_from_checkpoint(model, optim, lr_scheduler, args.saved_checkpoint, args.starting_epoch, 0)

    else:
        best_loss = 10000

    model.train()
    print('Training..')
    train (model,
           optim,
           lr_scheduler,
            out_name,
            test_loader,
            train_loader,
            MODELS_TRAIN_DIR,
            0,
            args,
            n_samples = n_samples,
            best_loss = best_loss)

    print('..Done')


############# Training main function #############
def train (model, optim, lr_scheduler, model_name, val_loader, data_loaders,
           saving_path, rank, args, n_samples, best_loss = 1e4):

    model.train()
    for epoch in range(args.starting_epoch, args.epochs + 1):
        mean_loss = 0

        for id, sample in enumerate (data_loaders):
            loss = model (sample)
            mean_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        if args.distributed:
            dist.barrier()

        # Test and save results of the model after each training epoch
        test_from_loader (val_loader, model, model_name, epoch)

        if epoch % args.save_epochs == 0:
            if rank == 0:
                print ('Loss: ', (mean_loss / n_samples))
                if (mean_loss / n_samples) < best_loss:
                    best_loss = mean_loss / n_samples

                    print(model_name, epoch, saving_path)
                if args.distributed:
                    save_checkpoint(model.module, optim, lr_scheduler, epoch, model_name, saving_path, best_loss)
                else:
                    save_checkpoint(model, optim, lr_scheduler, epoch, model_name, saving_path, best_loss)


############# Model testing after training epoch #############
def test_from_loader (val_data_loader, model, model_name, epoch):

    model.eval()
    if os.path.exists ("results/zuco/%s_%d.json"%(model_name, epoch)):
        os.remove ("results/zuco/%s_%d.json"%(model_name, epoch))

    results = []
    sample_id = 0
    for sample in val_data_loader:

        src_fmri = sample["signal"]

        ground_truth = sample["text_output"]
        output_text = model.generate (src_fmri)

        for a, b in  zip(output_text, ground_truth):
            results.append ({"text_predicted": a, "text_target": b})
            sample_id += 1

    with open("results/zuco/%s_%d.json"%(model_name, epoch), 'w') as out_file:
        json.dump(results, out_file)



############# Model testing #############
def test (model, results_fname, test_loader):
    print('Making datasets..')

    model.eval()

    results = []
    sample_id = 0
    for sample in test_loader:

        src_fmri = sample["signal"]
        ground_truth = sample["text_output"]
        output_text = model.generate (src_fmri)

        for a, b in  zip(output_text, ground_truth):
            results.append ({"text_predicted": a, "text_target": b})
            sample_id += 1

    if os.path.exists (f"results/zuco/{results_fname}.json"):
        os.remove (f"results/zuco/{results_fname}.json")

    with open(f"results/zuco/{results_fname}.json", 'w') as out_file:
        json.dump(results, out_file)



############# Checkpoint saving #############
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
        #"lr_scheduler": lr_scheduler.state_dict(),
        "lr_scheduler": None,
        'loss': loss,
        "epoch": cur_epoch}

    save_to = "%s/%s_epoch-%s.pth"%(saving_path, model_name, ("best" if is_best else str(cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


############# Checkpoint loading #############
def load_from_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, starting_epoch, rank=0):
    checkpoint = torch.load(checkpoint_path, map_location="cuda:%d"%rank)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)

    if "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "epoch" in checkpoint.keys():
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = starting_epoch +1

    if "loss" in checkpoint.keys():
        best_loss = checkpoint["loss"]
    else:
        best_loss = 1e4

    return start_epoch, best_loss
