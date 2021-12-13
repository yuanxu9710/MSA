import torch
from torch import nn
import torch.nn.functional as F

class TFN(nn.Module):
    def __init__(self, args):
        """
            Here we implement TFN
            Paper: Tensor Fusion Network for Multimodal Sentiment Analysis (EMNLP2017)
            Link: https://doi.org/10.18653/v1/d17-1115
        """
        super(TFN, self).__init__()

        ### input dimention related
        self.input_d_l, self.input_d_a, self.input_d_v = args.input_d_l, args.input_d_a, args.input_d_v
        self.lstm_d_l, self.lstm_d_a, self.lstm_d_v = args.lstm_d_l, args.lstm_d_a, args.lstm_d_v

        ### lstm related
        self.emb_dropout = args.emb_dropout
        self.lstm_net_l = nn.LSTM(self.input_d_l,  self.lstm_d_l)
        self.lstm_net_a = nn.LSTM(self.input_d_a,  self.lstm_d_a)
        self.lstm_net_v = nn.LSTM(self.input_d_v,  self.lstm_d_v)
        self.lstm_dropout_l, self.lstm_dropout_a, self.lstm_dropout_v = args.lstm_dropout_l, args.lstm_dropout_a, args.lstm_dropout_v
        self.lstm_postnet_l = nn.Sequential(nn.Dropout(self.lstm_dropout_l),
                                            nn.Linear(self.lstm_d_l, self.lstm_d_l),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_a = nn.Sequential(nn.Dropout(self.lstm_dropout_a),
                                            nn.Linear(self.lstm_d_a, self.lstm_d_a),
                                            nn.ReLU()
                                            )
        self.lstm_postnet_v = nn.Sequential(nn.Dropout(self.lstm_dropout_v),
                                            nn.Linear(self.lstm_d_v, self.lstm_d_v),
                                            nn.ReLU()
                                            )

    
        ### output related
        self.output_dropout = args.output_dropout
        self.output_mid_dim = args.output_mid_dim
        self.output_dim = args.output_dim
        self.output_layer = nn.Sequential(nn.Dropout(self.output_dropout),
                                        nn.Linear((self.lstm_d_l+1)*(self.lstm_d_a+1)*(self.lstm_d_v+1), self.output_mid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_mid_dim, self.output_mid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.output_mid_dim, self.output_dim),
                                        nn.ReLU()
                                        )

        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

 
    def forward(self, x_l, x_a, x_v):
        
        ### origin input: [batch_size, seq_len, n_features]
        ### lstm input and output: [seq_len, batch_size, n_features]
        x_l = F.dropout(x_l.transpose(0, 1), p=self.emb_dropout, training=self.training)
        x_a = x_a.transpose(0, 1)
        x_v = x_v.transpose(0, 1)

        ### [batch_size, hidden_dimention]
        _, lstm_h_l = self.lstm_net_l(x_l)
        h_l = lstm_h_l[0].squeeze()
        _, lstm_h_a = self.lstm_net_a(x_a)
        h_a = lstm_h_a[0].squeeze()
        _, lstm_h_v = self.lstm_net_v(x_v)
        h_v = lstm_h_v[0].squeeze()
        h_l = self.lstm_postnet_l(h_l)
        h_a = self.lstm_postnet_a(h_a)
        h_v = self.lstm_postnet_v(h_v)

        ### next we do tfn
        ### [batch_size, hidden_dimention+1]
        batch_size = h_l.size(0)
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(h_l).to(h_l.device)
        h_l = torch.cat((add_one, h_l), dim=1)
        h_a = torch.cat((add_one, h_a), dim=1)
        h_v = torch.cat((add_one, h_v), dim=1)

        ### [batch_size, hidden_dimention+1, 1] * [batch_size, 1, hidden_dimention+1] - > [batch_size, hidden_dimention+1, hidden_dimention+1]
        ###                                                                             > [batch_size, (hidden_dimention+1)*(hidden_dimention+1)]
        fusion_tensor = torch.bmm(h_l.unsqueeze(2), h_a.unsqueeze(1)).view(batch_size, -1)
        ### [batch_size, (hidden_dimention+1)*(hidden_dimention+1)] * [batch_size, 1, hidden_dimention+1] - > [batch_size, (hidden_dimention+1)*(hidden_dimention+1), hidden_dimention+1]
        ###                                                                             > [batch_size, (hidden_dimention+1)*(hidden_dimention+1)*(hidden_dimention+1)]
        fusion_tensor = torch.bmm(fusion_tensor.unsqueeze(2), h_v.unsqueeze(1)).view(batch_size, -1)
        ### relu - > (0, 1) - > (-3, 3)

        output = self.output_layer(fusion_tensor)
        output = torch.sigmoid(output) * self.output_range + self.output_shift
        return output, fusion_tensor


