import os
import argparse
import torch
from util.tool import *

torch.set_default_tensor_type('torch.FloatTensor')


### !!!YOU MAY SET IT FOR YOUR OWN EXPERIMENTS!!!
project_path = '/home/NewBio/xuyuan/btp/MSA'

parser = argparse.ArgumentParser(description='A set of experiments for MSA!')

### Model and Data
parser.add_argument('--model', type=str, default='misa', choices=['lf_lstm', 'ef_lstm', 'mult', 'tfn', 'misa'], help="set the model to run for MSA, choices=['mult', 'tfn', 'misa']")
parser.add_argument('--dataset', type=str, default='mosei', choices=['mosei', 'mosi'], help="set the dataset to run for MSA, choices=['mosei', mosi]")

### Data, Save and Log Path
parser.add_argument('--datapath', type=str, default=project_path+'/data', help="set the data path")
parser.add_argument('--savepath', type=str, default=project_path+'/save', help="set the model saving path")
parser.add_argument('--logpath', type=str, default=project_path+'/log', help="set the log path")

### GPU 
parser.add_argument('--seed', type=int, default=1234, help="set the random seed")
parser.add_argument('--use_gpu', type=bool, default=True, help="set whether use gpu")
parser.add_argument('--gpu_id', type=str, default='1', help="set the free gpu, right now only support for using one gpu")

### Training
parser.add_argument('--training', type=bool, default=True, help="set if training")
parser.add_argument('--optimizer', type=str, default='Adam', help="set the opt")
parser.add_argument('--criterion', type=str, default='L1Loss', help="set the criterion")

### Tuning
parser.add_argument('--batch_size', type=int, default=32, help="set the size of eath batch")
parser.add_argument('--epoch', type=int, default=100, help="set the size of iteration times")
parser.add_argument('--lr', type=float, default=5e-4, help="set the learning rate")


if __name__  == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    assign(args)    