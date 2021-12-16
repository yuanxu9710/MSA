class MISA_Config(object):
    def __init__(self, args):
        super(MISA_Config, self).__init__()
        self.args = args

    def config(self):
        self.args.input_d_l=300
        self.args.input_d_a=74
        self.args.input_d_v=35
        self.args.lstm_d_l=64
        self.args.lstm_d_a=32
        self.args.lstm_d_v=32
        self.args.emb_dropout=0.25
        self.args.project_dim=32
        self.args.encoder_dim=32
        self.args.num_heads=4
        self.args.layers=1
        self.args.output_dim=1
        self.args.dif_weight=0.3
        self.args.sim_weight=1.0
        self.args.rec_weight=1.0
        return self.args
        