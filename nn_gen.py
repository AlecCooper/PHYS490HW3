# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 3
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import torch as torch
import torch.nn as nn
import torch.nn.functional as func

class RBM(nn.Module):

    ## Arciecture:
    ## Two fully connected layers fc1 and fc2

    def __init__(self,x):
        super(RBM, self).__init__()

        # Calculate number of visible nodes
        num_v = x.shape[0]

        # Number of hidden nodes is the same as the number of visible node
        num_h = num_v

        # Initalize our weights and biases to random numbers
        self.w = torch.randn(num_v,num_h,dtype=torch.float64)
        self.w.requrires_grad = True
        self.c = torch.randn(1,num_v,dtype=torch.float64)
        self.c.requrires_grad = True
        self.b = torch.randn(1,num_h,dtype=torch.float64)
        self.b.requrires_grad = True

        # Create our two layers
        self.vl = x
        self.hl = torch.mv(self.w,self.vl) + self.b

    # Compute probability of v given h
    def prob_v(self,x):

        prob = torch.mv(self.w,x) + self.c
        prob = torch.sigmoid(prob)
        return(prob)

    # Compute probability of h given v
    def prob_h(self,x):
        
        prob = torch.m(x.t(),self.w) + self.b
        prob = torch.sigmoid(prob)
        return(prob)

    # computes the kl divergence
    def kl_div_weight(self,x,v):

        kl = torch.sum(torch.sum(torch.dot(self.prob_h(x),x))).item()/x.shape[0]
        
        kl += -torch.sum(torch.sum(torch.dot(self.prob_h(x),v))).item()/v.shape[0]

        print(kl)
        return kl

    # Feedforward functions
    def forward_hidden(self,x,v,lr):
        
        # Update the weights
        self.w += lr*(self.kl_div_weight(x,v))

        # Update the bias
        #self.b += lr * (x - v)

        # calculate the new hidden layer
        self.h = torch.mm(v,self.w) + self.c

    def forward_visible(self,phx,phv,x,v,lr):
        
        # Update the weights
        self.w += lr*(torch.mm(phx,x) - torch.mm(phv,v))

        # Update the bias
        self.b += lr * (x - v)




