import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pdb

class AttributeNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size, SEED):
        super(AttributeNetwork, self).__init__()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)

        self.Sfc1 = nn.Linear(input_size,hidden_size)
        self.Sdropout = nn.Dropout(p=0.3)
        self.Sfc2 = nn.Linear(hidden_size,output_size)

    def forward(self, x, y, labels):
        num_classes = x.shape[0]
        batch_size = y.shape[0]

        # the networks for attributes
        x = F.relu(self.Sfc1(x))
        x = self.Sdropout(x)
        x_processed = self.Sfc2(x)

        # matching networks
        output = torch.mm(y, x_processed.transpose(1,0))
        # norm1 = y.norm(2,1, keepdim=True).repeat(1, num_classes)
        # norm2 = x_processed.norm(2,1, keepdim=True).transpose(1,0).repeat(batch_size, 1)
        # pdb.set_trace()
        # output = output / norm1
        # output = output / norm2
        return output, x_processed