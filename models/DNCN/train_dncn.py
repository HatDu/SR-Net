import torch
import os

from models.DNCN.model_dncn import DnCn
from utils.train import build_optim, init_model, run_epoch, save_model, set_seeds
from utils.args import Args
from dataset import create_train_loaders
from tensorboardX import SummaryWriter

def build_model(args):
    model = DnCn(
        nc=args.nc,
        nd=args.nd,
        nf=args.nf
    ).to(args.device)
    init_model(model)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def main(args):
    exp_dir = args.exp_dir
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    writer = SummaryWriter(os.path.join(exp_dir, 'summary'))


    model = build_model(args)
    param_num = sum(param.numel() for param in model.parameters())
    # print(str(model))
    print('model params num %d'%param_num)

    data_loaders = create_train_loaders(args)
    print('train on %d samples, eval on %d samples'%(len(data_loaders['train']), len(data_loaders['dev'])))


    optimizer = build_optim(args, model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    
    start_epoch = 0
    best_dev_loss = 1e9
    for epoch in range(start_epoch, args.num_epochs):
        print('epoch %d'%epoch)
        scheduler.step(epoch)
        train_loss, dev_loss = run_epoch(args, epoch, model, data_loaders, optimizer, writer)
        save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, True)
        if dev_loss > best_dev_loss:
            best_dev_loss = dev_loss
            save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, True)
        else:
            save_model(args, exp_dir, epoch, model, optimizer, dev_loss, False)



def parse_args():
    parser = Args()
    parser.add_argument('-nc', type=int, default=5)
    parser.add_argument('-nd', type=int, default=5)
    parser.add_argument('-nf', type=int, default=32)
     
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    main(args)
    pass
