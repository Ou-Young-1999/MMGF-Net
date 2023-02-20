from torch import nn
import torch
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.dropout = dropout  
        self.alpha = alpha 
        self.concat = concat  

        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414) 
        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)  
        self.W3 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)  

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  
        adj:  [N, N] 
        """
        h1 = torch.matmul(inp, self.W1)  # [B, N, out_features]
        h2 = torch.matmul(inp, self.W2).permute(0,2,1)
        h3 = torch.matmul(inp, self.W3)

        e = self.leakyrelu(torch.matmul(h1,h2))

        zero_vec = -1e12 * torch.ones_like(e) 

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        attention = F.softmax(attention, dim=-1) 

        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        h_prime = torch.matmul(attention, h3)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class corGAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super(corGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention) 
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training) 
        # print(x.shape)
        x = F.elu(self.out_att(x, adj)) 
        return x
