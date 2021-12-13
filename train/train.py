from train.mult import train_mult
from train.tfn import train_tfn
from train.misa import train_misa
from train.ef_lstm import train_ef_lstm
from train.lf_lstm import train_lf_lstm

train_map = {'ef_lstm': train_ef_lstm, 'lf_lstm': train_lf_lstm, 'tfn': train_tfn, 'mult': train_mult, 'misa': train_misa}

def train(args, model, train_loader, valid_loader, test_loader):
    train_map[args.model](args, model, train_loader, valid_loader, test_loader)
