#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:14:43 2020

@author: angelica_cyj
"""
import torch
#import torch.nn.functional as F
import torch.nn as nn
#import numpy as np
#from torch.autograd import Variable
import math

test_t = torch.rand([2,1,20187])
f2 = test_t.view(-1,20187)
f1 = torch.tensor([224,5453]).view(-1,1)
final_t = test_t.view(20187,-1)

trg = torch.zeros(20187,dtype = torch.long)

t_idx = [34,65,12]

for index in t_idx:
    trg[index] = 1
    
softmax = nn.Softmax(dim=-1)

flatten = final_t.view(-1,1)
output = softmax(flatten)
topv, topi = output.data.topk(1,dim = 0)


d_model = 100
max_len = 200
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)

x = torch.randn([3,1,100])

add =pe[:x.size(0), :]
x = x + add
#
#sz = 6
#mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#
#mask = (torch.triu(torch.ones(10, 10)) == 1).transpose(0, 1)
#mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#nopeak_mask = np.triu(np.ones((1,10,10)),1).astype('uint8')
#nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
#target_msk = mask & nopeak_mask