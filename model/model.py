from model.mult import MulT
from model.tfn import TFN
from model.misa import MISA
from model.ef_lstm import EF_LSTM
from model.lf_lstm import LF_LSTM

model_map = {'ef_lstm': EF_LSTM, 'lf_lstm': LF_LSTM, 'tfn': TFN, 'mult': MulT, 'misa': MISA}

def get_model(args):
    return model_map[args.model](args)