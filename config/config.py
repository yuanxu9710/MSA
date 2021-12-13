from config.mult import MulT_Config
from config.tfn import TFN_Config
from config.misa import MISA_Config
from config.ef_lstm import EF_LSTM_Config
from config.lf_lstm import LF_LSTM_Config

config_map = {'ef_lstm': EF_LSTM_Config,'lf_lstm': LF_LSTM_Config, 'tfn': TFN_Config, 'mult': MulT_Config, 'misa': MISA_Config}

def get_config(args):
    return config_map[args.model](args).config()