class MulT_Config(object):
    def __init__(self, args):
        super(MulT_Config, self).__init__()
        self.args = args

    def config(self):
        self.args.input_d_l=300
        self.args.input_d_a=74
        self.args.input_d_v=35
        self.args.conv_d_l=30
        self.args.conv_d_a=30 
        self.args.conv_d_v=30
        self.args.output_dim=1
        self.args.num_heads=5
        self.args.layer=5
        self.args.attn_dropout=0.1
        self.args.relu_dropout=0.1
        self.args.res_dropout=0.1
        self.args.emb_dropout=0.25
        self.args.attn_mask=False
        self.args.output_drop=0
        return self.args
        