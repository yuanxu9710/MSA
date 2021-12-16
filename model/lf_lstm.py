import torch
from torch import nn
import torch.nn.functional as F

class LF_LSTM(nn.Module):
    def __init__(self, args):
        """
            Here we implement Late Fusion LSTM
        """
        super(LF_LSTM, self).__init__()

        ### input dimention related
        self.input_d_l, self.input_d_a, self.input_d_v = args.input_d_l, args.input_d_a, args.input_d_v
        self.lstm_d_l, self.lstm_d_a, self.lstm_d_v = args.lstm_d_l, args.lstm_d_a, args.lstm_d_v

        ### lstm related
        self.emb_dropout = args.emb_dropout
        self.lstm_net_l = nn.LSTM(self.input_d_l,  self.lstm_d_l, batch_first=True)
        self.lstm_net_a = nn.LSTM(self.input_d_a,  self.lstm_d_a, batch_first=True)
        self.lstm_net_v = nn.LSTM(self.input_d_v,  self.lstm_d_v, batch_first=True)
        self.lstm_postnet_l = nn.Sequential(nn.LayerNorm(self.lstm_d_l),
                                            nn.Linear(self.lstm_d_l, self.lstm_d_l),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_a = nn.Sequential(nn.LayerNorm(self.lstm_d_a),
                                            nn.Linear(self.lstm_d_a, self.lstm_d_a),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_v = nn.Sequential(nn.LayerNorm(self.lstm_d_v),
                                            nn.Linear(self.lstm_d_v, self.lstm_d_v),
                                            nn.ReLU()
                                            )
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
                                        )


 
    def forward(self, x_l, x_a, x_v):
        
        batch_size = x_l.size(0)
        ### origin input: [batch_size, seq_len, n_features]
        x_l = F.dropout(x_l, p=self.emb_dropout, training=self.training)
        x_a = x_a
        x_v = x_v

        ### [batch_size, seq_len, hidden_dimention]
        h_l, _ = self.lstm_net_l(x_l)
        h_a, _ = self.lstm_net_a(x_a)
        h_v, _ = self.lstm_net_v(x_v)
        h_l = self.lstm_postnet_l(h_l)
        h_a = self.lstm_postnet_a(h_a)
        h_v = self.lstm_postnet_v(h_v)

        ### late fusion
        h_all = torch.cat((h_l, h_a, h_v), dim=2)
        _, (h_fusion, _) = self.lstm_net_fusion(h_all)
        h = self.lstm_postnet_fusion(h_fusion.squeeze())

        ### output
        output = self.output_layer(h)
        return output


