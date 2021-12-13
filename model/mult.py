import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

class MulT(nn.Module):
    def __init__(self, args):
        """
            Here we implement Mult
            Paper: Multimodal Transformer for Unaligned Multimodal Language Sequences (ACL2019)
            Link: https://doi.org/10.18653/v1/p19-1656
        """
        super(MulT, self).__init__()

        ### dimention related
        self.input_d_l, self.input_d_a, self.input_d_v = args.input_d_l, args.input_d_a, args.input_d_v
        self.conv_d_l, self.conv_d_a, self.conv_d_v = args.conv_d_l, args.conv_d_a, args.conv_d_v
        self.output_dim = args.output_dim

        ### convolution related
        self.conv_net_l = nn.Conv1d(self.input_d_l, self.conv_d_l, kernel_size=1, padding=0, bias=False)
        self.conv_net_a = nn.Conv1d(self.input_d_a, self.conv_d_a, kernel_size=1, padding=0, bias=False)
        self.conv_net_v = nn.Conv1d(self.input_d_v, self.conv_d_v, kernel_size=1, padding=0, bias=False)

        ### transformer related
        self.num_heads = args.num_heads
        self.layers = args.layer
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.emb_dropout = args.emb_dropout
        self.attn_mask = args.attn_mask
        self.cross_trans_la = TransformerEncoder(self.conv_d_a, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.cross_trans_lv = TransformerEncoder(self.conv_d_v, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.cross_trans_al = TransformerEncoder(self.conv_d_l, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.cross_trans_av = TransformerEncoder(self.conv_d_v, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.cross_trans_vl = TransformerEncoder(self.conv_d_l, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.cross_trans_va = TransformerEncoder(self.conv_d_a, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.self_trans_av = TransformerEncoder(2*self.conv_d_l, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.self_trans_vl = TransformerEncoder(2*self.conv_d_a, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)
        self.self_trans_la = TransformerEncoder(2*self.conv_d_v, self.num_heads, self.layers, self.attn_dropout, self.relu_dropout, self.res_dropout, self.emb_dropout, self.attn_mask)

        ### linear related
        self.output_drop = args.output_drop
        self.linear1 = nn.Linear(2 * (self.conv_d_a + self.conv_d_l + self.conv_d_v), 2 * (self.conv_d_a + self.conv_d_l + self.conv_d_v))
        self.linear2 = nn.Linear(2 * (self.conv_d_a + self.conv_d_l + self.conv_d_v), 2 * (self.conv_d_a + self.conv_d_l + self.conv_d_v))
        self.output = nn.Linear(2 * (self.conv_d_a + self.conv_d_l + self.conv_d_v), self.output_dim)

    def forward(self, x_l, x_a, x_v):
        
        ### origin input: [batch_size, seq_len, n_features]
        ### cnn input and output: [batch_size, n_features, seq_len]
        x_l = F.dropout(x_l.transpose(1, 2), p=self.emb_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        ### [S, N, D]
        conv_x_l = self.conv_net_l(x_l).permute(2, 0, 1)
        conv_x_a = self.conv_net_a(x_a).permute(2, 0, 1)
        conv_x_v = self.conv_net_v(x_v).permute(2, 0, 1)

        h_l_a = self.cross_trans_la(conv_x_l, conv_x_a, conv_x_a)
        h_l_v = self.cross_trans_lv(conv_x_l, conv_x_v, conv_x_v)
        h_l = self.self_trans_av(torch.cat([h_l_a, h_l_v], dim=2))
        if type(h_l) == tuple:
            h_l = h_l[0]
        last_h_l = h_l[-1]

        h_a_v = self.cross_trans_av(conv_x_a, conv_x_v, conv_x_v)
        h_a_l = self.cross_trans_al(conv_x_a, conv_x_l, conv_x_l)
        h_a = self.self_trans_vl(torch.cat([h_a_v, h_a_l], dim=2))
        if type(h_a) == tuple:
            h_a = h_a[0]
        last_h_a = h_a[-1]

        h_v_l = self.cross_trans_vl(conv_x_v, conv_x_l, conv_x_l)
        h_v_a = self.cross_trans_va(conv_x_v, conv_x_a, conv_x_a)
        h_v = self.self_trans_la(torch.cat([h_v_l, h_v_a], dim=2))
        if type(h_v) == tuple:
            h_v = h_v[0]
        last_h_v = h_v[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        last_hs1 = self.linear2(F.dropout(F.relu(self.linear1(last_hs)), p=self.output_drop, training=self.training))
        last_hs2 = last_hs1+last_hs

        output = self.output(last_hs2)

        return output, last_hs


