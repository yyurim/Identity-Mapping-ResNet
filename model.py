import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

class StageBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, option=False):
        super(StageBlock, self).__init__()
        self.option = option

        if self.option:
            hidden_channel = input_channel*2
            self.option_b = nn.Conv2d(input_channel, hidden_channel, kernel_size=1, stride=2,padding=0)
        else:
            hidden_channel = input_channel
        

        self.batch_norm_1 = nn.BatchNorm2d(input_channel)

        self.conv_layer_1 = nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding)

        if self.option:
            stride = 1

        self.batch_norm_2 = nn.BatchNorm2d(hidden_channel)

        self.conv_layer_2 = nn.Conv2d(hidden_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    
    def forward(self,x):
        bn_1 = self.batch_norm_1(x)
        act_1 = F.relu(bn_1)
        conv_1 = self.conv_layer_1(act_1)

        bn_2 = self.batch_norm_2(conv_1)
        act_2 = F.relu(bn_2)
        conv_2 = self.conv_layer_2(act_2)

        if self.option:
            opt_b = self.option_b(act_1)
            out = torch.add(opt_b,conv_2)
        else:
            out = torch.add(x,conv_2)
        
        return out



class IdentityResNet(nn.Module):
    
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()

        self.conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.stage_1 = nn.Sequential(
            StageBlock(input_channel=64,output_channel=64,kernel_size=3,stride=1,padding=1),
            StageBlock(input_channel=64,output_channel=64,kernel_size=3,stride=1,padding=1)
        )

        self.stage_2 = nn.Sequential(
            StageBlock(input_channel=64,output_channel=128,kernel_size=3,stride=2,padding=1,option=True),
            StageBlock(input_channel=128,output_channel=128,kernel_size=3,stride=1,padding=1)
        )

        self.stage_3 = nn.Sequential(
            StageBlock(input_channel=128,output_channel=256,kernel_size=3,stride=2,padding=1,option=True),
            StageBlock(input_channel=256,output_channel=256,kernel_size=3,stride=1,padding=1)
        )

        self.stage_4 = nn.Sequential(
            StageBlock(input_channel=256,output_channel=512,kernel_size=3,stride=2,padding=1,option=True),
            StageBlock(input_channel=512,output_channel=512,kernel_size=3,stride=1,padding=1)
        )

        self.fc = nn.Linear(512,10)

        
    
    def forward(self, x):
        conv = self.conv_layer(x)

        s_1 = self.stage_1(conv)
        s_2 = self.stage_2(s_1)
        s_3 = self.stage_3(s_2)
        s_4 = self.stage_4(s_3)
        
        act = F.avg_pool2d(s_4, 4, 4)
        act = act.reshape(act.shape[0],-1)
        out = self.fc(act)
        return out