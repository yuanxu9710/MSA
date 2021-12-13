class TFN_Config(object):
    def __init__(self, args):
        super(TFN_Config, self).__init__()
        self.args = args

    def config(self):
        self.args.input_d_l=300
        self.args.input_d_a=74
        self.args.input_d_v=35
        self.args.lstm_d_l=32
        self.args.lstm_d_a=32 
        self.args.lstm_d_v=16
        self.args.lstm_dropout_l=0.1
        self.args.lstm_dropout_a=0.1
        self.args.lstm_dropout_v=0.1
        self.args.output_dropout=0.1
        self.args.output_mid_dim=32
        self.args.output_dim=1
        self.args.emb_dropout=0.25
        return self.args
        