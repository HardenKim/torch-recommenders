import torch
import torch.nn as nn

class GCN_Layer(nn.Module):
    """_summary_
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_matrix = adj_matrix
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        out = self.fc(torch.spmm(self.adj_matrix, x)) # 이웃 정보 종합
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

        
class GCN(nn.Module):
    def __init__(self, num_feature, num_hidden, num_class, adj_matrix, dropout):
        super(GCN, self).__init__()
        self.layer = nn.Sequential(GCN_Layer(num_feature, num_hidden, adj_matrix),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   GCN_Layer(num_hidden, num_class, adj_matrix))
        
    def forward(self, x):
        out = self.layer(x)
        return out
    
    def __repr__(self):
        return self.__class__.__name__

 
class FCN(nn.Module):
    def __init__(self, num_feature, num_hidden, num_class, dropout):
        super(FCN, self).__init__()
        self.layer = nn.Sequential(nn.Linear(num_feature, num_hidden),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(num_hidden, num_class))
        
    def forward(self, x):
        out = self.layer(x)
        return out
    
    def __repr__(self):
        return self.__class__.__name__
