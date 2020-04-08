import argparse

import torch
import os

# from utils.run import run_net
from dataset import create_test_loader
from tensorboardX import SummaryWriter
from models.CRNN.train_crnn import build_model
from utils.run_2d import run_net

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    return model


def main(args):
    model = load_model(args.ckpt)
    data_loader = create_test_loader(args)
    run_net(args, model, data_loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', type=str, required=True)
    parser.add_argument('-out-dir', type=str, default='../data/infer/')
    parser.add_argument('-data-path', type=str, default='../data/CC359/')
    parser.add_argument('-mask-style', default='cartesian_1d', type=str)
    parser.add_argument('-gap', default=1, type=int, help='sequence sample interval')
    parser.add_argument('-same', action='store_true', help='all mask will be same in a sequence')
    parser.add_argument('-cf', default=0.08, type=float, help='center fraction')
    parser.add_argument('-acc', default=4, type=float, help='accerlation')
    
    parser.add_argument('-dset', default='calgary')
    parser.add_argument('-resolution', default=320, type=int)
    parser.add_argument('-acquisition', nargs='+', default=['CORPD_FBK', 'CORPDFS_FBK'])

    parser.add_argument('-device', type=str, default='cuda')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
