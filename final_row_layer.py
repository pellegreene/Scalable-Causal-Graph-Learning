#Learning Temporal Nonlinearity, the front row inputs
# Using PyTorch and Python
#Here are your notes in case you forget about this stuff

#Python Imports
import math

#Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#Package Imports
import numpy as np
import pandas as pd

class final_row_model(nn.Module):
    def __init__(self, args, data) -> None:
        super(final_row_model, self).__init__()
        #get all parameters
        self.window = args.pre_win
        self.m = data.m
        self.p = (args.p_list) 
        self.len_p = len(args.p_list) 
        self.compress_p = args.compress_p_list
        self.p_sum = np.sum(self.p_list)
        self.len_compress_p = len(self.compress_p)
        self.cuda = args.cuda

        if self.len_compress_p>0:
            self.compress_p = args.compress_p[-1]
            self.weight = nn.Parameter(torch.ones([self.m, self.compress_p, self.window]))
        else:
            self.weight = nn.Parameter(torch.ones([self.m, self.p_sum, self.window]))
        
        #Using a kaiming uniform input here to increase the reliability of our initialization
        #A randomized model weight would could result in an unstable initialization, where
        #kaiming eliminates both the vanishing and exploding gradient problems
        #by giving us a uniform distribution of weights with mean incrementing layer by layer and std close to 1
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        #building a bias out of the input matrix and the window
        self.bias = Parameter(torch.Tensor(self.m,self.window)) 
        #calculating fan ins and outs for the kaiming inputs
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #Calculate what the bounds will be for our biases
        bound = 1 / math.sqrt(fan_in)
        #setting the bias on our model between bounds
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #Initialize a matrix of garbage values
        if self.window ==1:
            final_y = torch.empty(input.shape[0], self.m) 
        else:
            final_y = torch.empty(input.shape[0], self.window, self.m) 
        
        for j in range(self.m):           
            if self.window ==1:   
                
                final_y[:,j] = F.linear(input[:,j,:], self.weight[j,:].view(1, self.weight.shape[1]), self.bias[j,:]).view(-1);               
            else:
                
                final_y[:,:,j] = F.linear(input[:,j,:], self.weight[j,:].transpose(1,0), self.bias[j,:]);               
        
        
        
        if self.cuda:
            final_y = final_y.cuda()
        
        return final_y;
    
    def get_pi_weight(self):
        if self.len_compress_p>0:
            func_1 = nn.MaxPool1d(kernel_size=self.compress_p, stride=self.compress_p)
        else:
            func_1 = nn.MaxPool1d(kernel_size=self.p[0], stride=self.p[0])
        func_2 = nn.MaxPool1d(kernel_size=self.m, stride=self.m)
        
        weight1_norm_all = np.zeros((self.weight.shape[0], self.len_p))
        weight2_norm_all = np.zeros((self.len_p))
        for layer_i in range(self.weight.shape[-1]):
            weight_tmp = self.weight[:,:,layer_i]
            weight0 = weight_tmp.view(1, self.weight.shape[0],self.weight.shape[1])
            weight1 = func_1(torch.abs(weight0)) 
            weight1_inv = weight1.transpose(2,1).contiguous(); #mxp
            weight2 = func_2(weight1_inv).detach().numpy().ravel() 
            weight2_norm = weight2/np.sum(weight2)
            weight1_norm = F.normalize(weight1, p=1, dim=1).view(weight1.shape[1], weight1.shape[2]).detach().numpy()
            
            weight1_norm_all = weight1_norm_all + weight1_norm
            #pdb.set_trace()
            weight2_norm_all = weight2_norm_all + weight2_norm
        
        #pdb.set_trace()
        return weight1_norm_all, weight2_norm_all