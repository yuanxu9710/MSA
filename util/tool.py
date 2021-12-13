import os
import torch
from util.dataset import MSADataset
from torch.utils.data import DataLoader
from config.config import *
from model.model import *
from train.train import *


def get_data(args, split_type):
    datapath = os.path.join(args.datapath, args.dataset) + f'_{split_type}.g'
    if not os.path.exists(datapath):
        print("Auto generate data...")
        data = MSADataset(args.datapath, args.dataset, split_type)
        torch.save(data, datapath)
    else:
        print("Loading generated data cache...")
        data = torch.load(datapath)
    return data

def assign(args):
    # prepare data
    train_data = get_data(args, 'train')
    valid_data = get_data(args, 'valid')
    test_data = get_data(args, 'test')
    args.train_len = len(train_data)
    args.valid_len = len(valid_data)
    args.test_len = len(test_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # config model
    args = get_config(args)

    # instantiate model
    model = get_model(args)

    # train
    train(args, model, train_loader, valid_loader, test_loader)
