from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torch
from torch import nn
import shutil
import os
import numpy as np
import random

def set_seeds(seed):
    print('using seed: %d'%(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
            nn.init.xavier_uniform_(m.weight)

def build_optim(args, params):
    print('using RMSprop optimizer')
    # optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, oth=''):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=os.path.join(exp_dir, oth+'model.pt') 
    )
    if is_new_best:
        shutil.copyfile(os.path.join(exp_dir, 'model.pt') , os.path.join(exp_dir, 'best_model.pt') )

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    train_loss = 0.
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(train_loss=0.)]) as t:
        for iter, batch in enumerate(data_loader):
            data, _ = batch
            im_und, k_und, image, mask, kspace, pdf = data

            im_und = im_und.to(args.device).squeeze(0)
            k_und = k_und.to(args.device).squeeze(0)
            mask = mask.to(args.device).squeeze(0)
            image = image.to(args.device).squeeze(0)

            optimizer.zero_grad()
            im_und.requires_grad = True
            output = model(im_und, k_und, mask)

            loss = F.l1_loss(output, image)
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()

            writer.add_scalar('train_loss', loss.item(), global_step + iter)
            t.postfix[0]["train_loss"] = '%.4f' % (train_loss/(iter + 1))
            t.update()
        writer.add_scalar('TrainLoss', loss.item(), epoch)
        return train_loss/(iter + 1)

def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    eval_loss = 0.
    with tqdm(total=len(data_loader), postfix=[dict(eval_loss=0.)]) as t:
        with torch.no_grad():
            for iter, batch in enumerate(data_loader):
                data, _ = batch
                im_und, k_und, image, mask, kspace, pdf = data
                im_und = im_und.to(args.device).squeeze(0)
                k_und = k_und.to(args.device).squeeze(0)
                mask = mask.to(args.device).squeeze(0)
                image = image.to(args.device).squeeze(0)

                output = model(im_und, k_und, mask)
                
                eval_loss += (output - image).abs().mean()
                t.postfix[0]["eval_loss"] = '%.4f' % (eval_loss/(iter +1))
                t.update()
            writer.add_scalar('EvalLoss', eval_loss/(iter +1), epoch)
    return eval_loss/(iter +1)


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            data, _ = batch
            im_und, k_und, image, mask, kspace, pdf = data
            im_und = im_und.to(args.device).squeeze(0)
            k_und = k_und.to(args.device).squeeze(0)
            mask = mask.to(args.device).squeeze(0)
            image = image.squeeze(0)

            output = model(im_und, k_und, mask)

            output = (output**2).sum(-1).sqrt().cpu()
            target = (image**2).sum(-1).sqrt()

            n = 0
            step=target.size(0)//16
            choice = range(n, target.size(0), step)
            target = target[choice].unsqueeze(1)
            output = output[choice].unsqueeze(1)

            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image((target - output).abs(), 'Error')
            break

def run_epoch(args, epoch, model, data_loaders, optimizer, writer):
    train_loss = train_epoch(args, epoch, model, data_loaders['train'], optimizer, writer)
    if args.eval:
        dev_loss = evaluate(args, epoch, model, data_loaders['dev'], writer)
    else:
        dev_loss = train_loss
    if (epoch + 1) % args.log_interval == 0:
        visualize(args, epoch, model, data_loaders['display'], writer)
    torch.cuda.empty_cache()
    return train_loss, dev_loss