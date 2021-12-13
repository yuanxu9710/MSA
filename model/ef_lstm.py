import torch
from torch import nn
import torch.nn.functional as F

class EF_LSTM(nn.Module):
    def __init__(self, args):
        """
            Here we implement Late Fusion LSTM
        """
        super(EF_LSTM, self).__init__()

        ### input dimention related
        self.input_d_l, self.input_d_a, self.input_d_v = args.input_d_l, args.input_d_a, args.input_d_v
        self.lstm_d_l, self.lstm_d_a, self.lstm_d_v = args.lstm_d_l, args.lstm_d_a, args.lstm_d_v
        self.emb_dropout = args.emb_dropout

        ### linear related
        self.linear_l = nn.Sequential(nn.LayerNorm(self.input_d_l),
                                    nn.Linear(self.input_d_l, self.lstm_d_l),
                                    nn.ReLU())
        self.linear_a = nn.Sequential(nn.LayerNorm(self.input_d_a),
                                    nn.Linear(self.input_d_a, self.lstm_d_a),
                                    nn.ReLU())
        self.linear_v = nn.Sequential(nn.LayerNorm(self.input_d_v),
                                    nn.Linear(self.input_d_v, self.lstm_d_v),
                                    nn.ReLU())

        ### lstm related
        self.fusion_dim = args.fusion_dim
        self.lstm_net_fusion = nn.LSTM(self.lstm_d_l+self.lstm_d_a+self.lstm_d_v, self.fusion_dim, batch_first=True)
        self.lstm_postnet_fusion = nn.Sequential(nn.BatchNorm1d(self.fusion_dim),
                                                nn.Linear(self.fusion_dim, self.fusion_dim),
                                                nn.ReLU()
                                                )

        ### output related
        self.output_mid_dim = args.output_mid_dim
        self.output_dim = args.output_dim
        self.output_layer = nn.Sequential(nn.BatchNorm1d(self.fusion_dim),
                                        nn.Linear(self.fusion_dim, self.output_mid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_mid_dim, self.output_dim),
                                        nn.ReLU()
                                        )

        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

 
    def forward(self, x_l, x_a, x_v):
        
        batch_size = x_l.size(0)
        ### origin input: [batch_size, seq_len, n_features]
        x_l = F.dropout(x_l, p=self.emb_dropout, training=self.training)
        x_l = self.linear_l(x_l)
        x_a = self.linear_a(x_a)
        x_v = self.linear_v(x_v)

        x_fusion = torch.cat((x_l, x_a, x_v), dim=2)

        ### lstm
        _, (h, _) = self.lstm_net_fusion(x_fusion)
        h = self.lstm_postnet_fusion(h.view(batch_size, -1))

        ### output
        output = self.output_layer(h)
        output = torch.sigmoid(output) * self.output_range + self.output_shift
        return output


