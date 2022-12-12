import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.preprocessing import normalize
## granger full rank

class granger_model(nn.Module):
    def __init__(self, data, rank, cuda):
        self.cuda = cuda
        super(granger_model, self).__init__()
        self.weight = nn.Parameter(torch.ones([rank,1]))
        
        
        
    def forward(self, x):
        #l = x.shape[1]
        k = x.shape[2]
        
        y = torch.Tensor(x.shape)
        

        for j in range(k):
            
            tmp_new = torch.mul(x[:,:,j], self.weight[j,0])
            y[:,:,j] = tmp_new
            
        #pdb.set_trace()
        
        if torch.cuda.is_available():
            y = y.cuda()
        
        
        return y