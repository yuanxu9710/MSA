import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

class MISA(nn.Module):
    def __init__(self, args):
        """
            Here we implement MISA
            Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis (ACM2020)
            Link: https://dl.acm.org/doi/10.1145/3394171.3413678
        """ 
        super(MISA, self).__init__()

        ### input dimention related
        self.input_d_l, self.input_d_a, self.input_d_v = args.input_d_l, args.input_d_a, args.input_d_v
        self.lstm_d_l, self.lstm_d_a, self.lstm_d_v = args.lstm_d_l, args.lstm_d_a, args.lstm_d_v

        ### lstm related
        self.emb_dropout = args.emb_dropout
        self.lstm_net1_l = nn.LSTM(self.input_d_l,  self.lstm_d_l, batch_first=True, bidirectional=True)
        self.lstm_net1_a = nn.LSTM(self.input_d_a,  self.lstm_d_a, batch_first=True, bidirectional=True)
        self.lstm_net1_v = nn.LSTM(self.input_d_v,  self.lstm_d_v, batch_first=True, bidirectional=True)
        self.lstm_net2_l = nn.LSTM(self.lstm_d_l * 2,  self.lstm_d_l, batch_first=True, bidirectional=True)
        self.lstm_net2_a = nn.LSTM(self.lstm_d_a * 2,  self.lstm_d_a, batch_first=True, bidirectional=True)
        self.lstm_net2_v = nn.LSTM(self.lstm_d_v * 2,  self.lstm_d_v, batch_first=True, bidirectional=True)
        self.lstm_layernorm_l = nn.LayerNorm(self.lstm_d_l * 2)
        self.lstm_layernorm_a = nn.LayerNorm(self.lstm_d_a * 2)
        self.lstm_layernorm_v = nn.LayerNorm(self.lstm_d_v * 2)
        self.lstm_postnet_l = nn.Sequential(nn.BatchNorm1d(self.lstm_d_l * 4),
                                            nn.Linear(self.lstm_d_l * 4, self.lstm_d_l),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_a = nn.Sequential(nn.BatchNorm1d(self.lstm_d_a * 4),
                                            nn.Linear(self.lstm_d_a * 4, self.lstm_d_a),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_v = nn.Sequential(nn.BatchNorm1d(self.lstm_d_v * 4),
                                            nn.Linear(self.lstm_d_v * 4, self.lstm_d_v),
                                            nn.ReLU()
                                            )

        ### shared private related
        self.project_dim = args.project_dim
        self.encoder_dim = args.encoder_dim
        self.project_l = nn.Sequential(nn.BatchNorm1d(self.lstm_d_l),
                                            nn.Linear(self.lstm_d_l, self.project_dim),
                                            nn.ReLU()
                                            )
        self.project_a = nn.Sequential(nn.BatchNorm1d(self.lstm_d_a),
                                            nn.Linear(self.lstm_d_a, self.project_dim),
                                            nn.ReLU()
                                            )
        self.project_v = nn.Sequential(nn.BatchNorm1d(self.lstm_d_v),
                                            nn.Linear(self.lstm_d_v, self.project_dim),
                                            nn.ReLU()
                                            )
        self.share_encoder = nn.Sequential(nn.BatchNorm1d(self.project_dim),
                                            nn.Linear(self.project_dim, self.encoder_dim),
                                            nn.ReLU()
                                            )
        
        self.private_encoder_l = nn.Sequential(nn.BatchNorm1d(self.project_dim),
                                            nn.Linear(self.project_dim, self.encoder_dim),
                                            nn.ReLU()
                                            )
        self.private_encoder_a = nn.Sequential(nn.BatchNorm1d(self.project_dim),
                                            nn.Linear(self.project_dim, self.encoder_dim),
                                            nn.ReLU()
                                            )
        self.private_encoder_v = nn.Sequential(nn.BatchNorm1d(self.project_dim),
                                            nn.Linear(self.project_dim, self.encoder_dim),
                                            nn.ReLU()
                                            )

        ### reconstruct net
        self.recon_net_l = nn.Sequential(nn.Linear(self.encoder_dim, self.lstm_d_l))
        self.recon_net_a = nn.Sequential(nn.Linear(self.encoder_dim, self.lstm_d_a))
        self.recon_net_v = nn.Sequential(nn.Linear(self.encoder_dim, self.lstm_d_v))

        ### output related
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.output_dim = args.output_dim
        self.fusion = nn.Sequential(nn.BatchNorm1d(self.encoder_dim*6),
                                    nn.Linear(self.encoder_dim*6, self.encoder_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.encoder_dim, self.output_dim),
                                    )
        self.transformer = TransformerEncoder(self.encoder_dim, self.num_heads, self.layers)
        # self.range = nn.Parameter(torch.FloatTensor([6]),requires_grad=False)
        # self.shift = nn.Parameter(torch.FloatTensor([-3]),requires_grad=False)

 
    def forward(self, x_l, x_a, x_v):
        
        ### origin input: [batch_size, seq_len, n_features]
        x_l = F.dropout(x_l, p=self.emb_dropout, training=self.training)
        batch_size = x_l.size(0)

        ### sLSTM
        ### h1 : [batch_size, seq_len, 2*hidden_dimention]
        ### final_h1 : [2, batch_size,  hidden_dimention]
        ### final_h2 : [2, batch_size, hidden_dimention]
        ### h : [batch_size,  hidden_dimention]
        h1_l, (final_h1_l, _) = self.lstm_net1_l(x_l)
        h1_a, (final_h1_a, _) = self.lstm_net1_a(x_a)
        h1_v, (final_h1_v, _)= self.lstm_net1_v(x_v)
        h1_l = self.lstm_layernorm_l(h1_l)
        h1_a = self.lstm_layernorm_a(h1_a)
        h1_v = self.lstm_layernorm_v(h1_v)
        _, (final_h2_l, _) = self.lstm_net2_l(h1_l)
        _, (final_h2_a, _) = self.lstm_net2_a(h1_a)
        _, (final_h2_v, _)= self.lstm_net2_v(h1_v)
        h_l = torch.cat((final_h1_l, final_h2_l), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        h_a = torch.cat((final_h1_a, final_h2_a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        h_v = torch.cat((final_h1_v, final_h2_v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        self.h_l = self.lstm_postnet_l(h_l)
        self.h_a = self.lstm_postnet_a(h_a)
        self.h_v = self.lstm_postnet_v(h_v)

        ### shared and private, we need to keep the nums to calculate the sim and dif loss
        ### shared and private: [batch_size, encoder_dim]
        pro_h_l = self.project_l(self.h_l)
        pro_h_a = self.project_a(self.h_a)
        pro_h_v = self.project_v(self.h_v)
        self.shared_h_l = self.share_encoder(pro_h_l)
        self.shared_h_a = self.share_encoder(pro_h_a)
        self.shared_h_v = self.share_encoder(pro_h_v)
        self.private_h_l = self.private_encoder_l(pro_h_l)
        self.private_h_a = self.private_encoder_a(pro_h_a)
        self.private_h_v = self.private_encoder_v(pro_h_v)

        ### reconstruct, we need to keep the nums to calculate the recon loss
        self.recon_h_l = self.recon_net_l(self.shared_h_l + self.private_h_l)
        self.recon_h_a = self.recon_net_a(self.shared_h_a + self.private_h_a)
        self.recon_h_v = self.recon_net_v(self.shared_h_v + self.private_h_v)
        
        ### transformer fusion
        h_all = torch.stack((self.shared_h_l, self.shared_h_a, self.shared_h_v, self.private_h_l, self.private_h_a, self.private_h_v), dim=0)
        h = self.transformer(h_all)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        h = self.fusion(h)
        output = h
        # output = self.range * torch.sigmoid(h) + self.shift

        return output


