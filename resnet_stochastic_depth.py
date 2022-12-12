from torchvision.models import resnet50
from torchvision.ops import stochastic_depth
import granger
import final_row_layer
from torch.nn import Sequential
import torch.nn
import torch.nn.utils.weight_norm
import torch.nn.functional

import numpy as np


#ResNet with Stochastic Depth, the first Module in the pipeline
class resnet_stochastic(torch.nn.Module):
    def __init__(self, data) -> None:
        super(resnet_stochastic).__init__()
        self.resnet = resnet50()
        self.data = data
        self.betas = [0.5, 0.5, 0.5, 0.5]
        self.counter = 0
        self.conv1 = torch.nn.Sequential(*list(self.resnet.children())[0:3])
        self.conv2 = torch.nn.Sequential(*list(self.resnet.children())[4])
        self.conv3 = torch.nn.Sequential(*list(self.resnet.children())[5])
        self.conv4 = torch.nn.Sequential(*list(self.resnet.children())[6])
        self.conv5 = torch.nn.Sequential(*list(self.resnet.children())[7:])

    def get_probs(self, current_layer, final_layer):
        b_Q = 0.5
        if self.counter < 4: 
            self.counter = self.counter + 1
            return 0.5
        new_p = 1-float(current_layer/final_layer)*(1-b_Q)
        self.betas[current_layer-1] = new_p
        return new_p
        
    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        out3 = stochastic_depth(input=out2, p=self.get_probs(1,4))
        out4 = self.conv3(out3)
        out5 = stochastic_depth(input=out4, p=self.get_probs(2,4))
        out6 = self.conv4(out5)
        out7 = stochastic_depth(input=out6, p=self.get_probs(3,4))
        out8 = self.conv4(out7)
        out9 = stochastic_depth(input=out8, p=self.get_probs(4,4))
        out10 = self.conv5(out9)

        approximator_outputs = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]
        return out10

class weight_model(torch.nn.Module):
    def __init__(self) -> None:
        super(weight_model).__init__()
        self.weight = torch.nn.Parameter(torch.ones([1]))

    def forward(self, input):
        return torch.mul(input, self.weight)

class causal_graph_approximation(torch.nn.Module):
    def __init__(self, args, approximator_inputs) -> None:
        self.beta_list = args.beta_list
        self.beta_sum = np.sum(args.beta_list)
        self.compressed_beta_list = args.compressed_beta_list
        self.x  = approximator_inputs
        self.m = approximator_inputs.m
        self.k_list = k_list
        self.new_k_list = [approximator_inputs.m]+self.k_list
        self.window = window

        self.P = []
        self.orthogonal_mat = []
        self.sparse_mat = []

        for _ in range(len(self.beta_list)):
            self.P.append(weight_model())
        self.P = torch.nn.ModuleList(self.P)

        self.linear_layers = [torch.nn.Linear(self.window, self.beta_list[0])]
        self.sparse_mat.append(0)
        self.orthogonal_mat.append(1)

        for i in range(1,len(self.beta_list)):
            for j in range(1,len(self.k_list)):
                self.linear_layers.append(torch.nn.utils.weight_norm(torch.nn.Linear(self.new_k_list[j-1], self.new_k_list[j])))
                self.sparse_mat.append(1)
                if j == 0:
                    self.orthogonal_mat.append(1)
                else: self.orthogonal_mat.append(2)

                self.linear_layers.append(torch.nn.BatchNorm1d(self.p_list[-1])); #m->k
                self.sparse_mat.append(0)
                self.orthogonal_mat.append(0)

            self.linear_layers.append(granger.granger_model(approximator_inputs, self.k_list[-1], self.use_cuda)) #k->k
            self.sparse_mat.append(0)
            self.orthogonal_mat.append(0)

            for j in range(1, len(self.k_list)):       
                self.linear_layers.append( torch.nn.utils.weight_norm(torch.nn.Linear(self.new_k_list[j], self.new_k_list[j-1], bias = False))); #m->m, supervised
                self.sparse_mat.append(1)
                if j == 1:
                    self.orthogonal_mat.append(1)
                else:
                    self.orthogonal_mat.append(2)
                self.linear_layers.append(torch.nn.BatchNorm1d(self.p_list[-1])); #m->k
                self.sparse_mat.append(0)
                self.orthogonal_mat.append(0)



            
        if len(self.compressed_beta_list)>0:
            self.linear_layers.append( (torch.nn.Linear(self.beta_sum, self.compressed_beta_list[0])))
            self.sparse_mat.append(0) 
            self.orthogonal_mat.append(1)
            
            for i in range(1,len(self.compressed_beta_list)):
                self.linear_layers.append( (torch.nn.Linear(self.compressed_beta_list[i-1], self.compressed_beta_list[i])))
                self.sparse_mat.append(0) 
                self.orthogonal_mat.append(1)
          
#        
        self.linear_layers.append(final_row_layer.final_row_model(args, approximator_inputs)); #k->k  
        self.sparse_mat.append(1)
        self.orthogonal_mat.append(0)
        
        
        
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.dropout = torch.nn.Dropout(args.dropout)

        for i in range(len(self.linear_layers)):
            if not isinstance(self.linear_layers[i], torch.nn.InstanceNorm1d) and not isinstance(self.linear_layers[i], torch.nn.BatchNorm1d) and not isinstance(self.linear_layers[i], granger.granger_model):
                W = self.linear_layers[i].weight.transpose(0,1).detach().numpy()
                ## sparsity
                if W.ndim >=2 and self.orthogonal_mat[i]==1: ## sparsity
                    #nn.init.xavier_normal_(self.linear_layers[i].weight)
                    self.linear_layers[i].weight = torch.nn.init.orthogonal_(self.linear_layers[i].weight)
                if W.ndim >=2 and self.orthogonal_mat[i]>1: ## sparsity
                    #nn.init.xavier_normal_(self.linear_layers[i].weight)
                    tmp = self.linear_layers[i].weight
                    self.linear_layers[i].weight = np.eye(tmp.shape[0],tmp.shape[1])


    def forward(self, inputs):
        x_input = inputs[0] #pxm 
        x = x_input.transpose(2,1).contiguous(); #mxp
        x = self.dropout(x)            
        x_org = x
        x_p = []

        if self.p_list[0]> self.w:
            padding = torch.nn.ConstantPad2d((0, self.p_list[0]-self.w, 0, 0), 0)
            x_0n = padding(x_org)
        
        x_0 = x_org
        for i in range(len(self.beta_list)): 
            x_i = self.linear_layers[i](x_0)
            x_i = torch.nn.functional.relu(self.P[i](x_i) + x_0n)
            x_0n = x_i
            x_0 = x_i
            x_p.append(x_i)
        
        x_p_m = []  
        for i in range(len(self.beta_list)):
            
            x_sp =  x_p[i].transpose(2,1).contiguous(); ## read the data piece  
            
            x_sp_tmp = []
            x_sp_tmp.append(x_sp)
            for j in range(len(self.k_list)):   
                x_sp = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+2*j](x_sp);  #lxk 
                x_sp = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+2*j+1](x_sp);  #lxk 
                x_sp = torch.nn.functional.tanh(x_sp/5.)
                x_sp = self.dropout(x_sp)
                x_sp_tmp.append(x_sp)
            
            x_sp = self.linear_layers[self.len(self.beta_list)+i*(4*len(self.k_list)+1) + 2*len(self.k_list)]  #lxk 
            
            for j in range(0,len(self.k_list)):  
                x_sp = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+2*len(self.k_list) + 1+2*j](x_sp);  #lxm
                x_sp = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+2*len(self.k_list) + 1+2*j+1](x_sp);  #lxm
                x_sp = torch.nn.functional.relu(x_sp/1.)
                x_sp = self.dropout(x_sp)
            
            x_sp = x_sp.transpose(2,1).contiguous(); #mxl
            x_p_m.append(x_sp)
            
        x_p_m = torch.cat(x_p_m, dim = 2) 
        
            
        if len(self.compressed_beta_list)>0:
            for j in range(len(self.compressed_beta_list)): 
                x_p_m = self.linear_layers[len(self.beta_list)+len(self.beta_list)*(4*len(self.k_list)+1)+j](x_p_m); #mx2
                x_p_m = torch.nn.functional.tanh(x_p_m/5.)
                x_sp = self.dropout(x_sp)
        
        final_y = self.linear_layers[-1](x_p_m)

        return final_y      

    def predict_relationship(self):
        CGraph_list1 = []
        CGraph_list2 = []

        G_1 = np.zeros((self.m,self.m))
        G_2 = np.zeros((self.m,self.m))
        G_3 = np.zeros((self.m,self.m))
        G_4 = np.zeros((self.m,self.m))
        
                
        for i in range(len(self.beta_list)):
            pl = self.P[i].weight.data
            
            A = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+0].weight.transpose(0,1).cpu().detach().numpy()
            B = np.diag(self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+2*len(self.k_list)].weight.transpose(0,1).detach().cpu().numpy().ravel())
            C = self.linear_layers[len(self.beta_list)+i*(4*len(self.k_list)+1)+4*len(self.k_list)+1-2].weight.transpose(0,1).cpu().detach().numpy()
            #CGraph1 = np.abs(np.dot(A,C))#
            CGraph1 = np.abs(np.dot(np.dot(A,B),C))
            CGraph1[range(self.m), range(self.m)] = 0    
            CGraph_list1.append(CGraph1)
            
            A = np.abs(A) 
            B = np.abs(B) 
            C = np.abs(C) 
            #CGraph2 = np.abs(np.dot(A,C))#
            CGraph2 = np.abs(np.dot(np.dot(A,B),C))
            CGraph2[range(self.m), range(self.m)] = 0    
            CGraph_list2.append(CGraph2)   
   
                
            G_1 = np.add(G_1, CGraph1)
            G_2 = np.add(G_2, CGraph2)
            
            G_3 = np.add(G_3, np.multiply(CGraph1, pl.cpu().detach().numpy())) 
            G_4 = np.add(G_4, np.multiply(CGraph2, pl.cpu().detach().numpy()))    
      
        G_1[range(self.m), range(self.m)] = 0 
        G_2[range(self.m), range(self.m)] = 0 
        G_3[range(self.m), range(self.m)] = 0 
        G_4[range(self.m), range(self.m)] = 0


resnet = resnet50()
if torch.cuda.is_available():
    resnet.cuda()
# torchsummary.summary(resnet,(3, 224,224))
child_counter=0
for child in resnet.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

k_list = [30] #Not specifically sure what this refers to, adding it because they do?
window = 3

