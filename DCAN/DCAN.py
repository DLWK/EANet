import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
from torchvision import models
import sys, os

# from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm = nn.BatchNorm2d


class DCANet(nn.Module):
    def __init__(self,n_channels, n_classes, abn=BatchNorm):
        super(DCANet, self).__init__()
	
        ################################vgg16#######################################
        feats = list(models.vgg19_bn(pretrained=False).features.children())

        feats[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   #适应任务
        self.conv1 = nn.Sequential(*feats[:6])
        # print(self.conv1)
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])
        self.conv6 = nn.Sequential (
                    nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU()
                )

        ################################Gate#######################################
        self.out =  nn.Conv2d(2048, 1, kernel_size=1)
        self.out1 =  nn.Conv2d(2, 1, kernel_size=1)
    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)   #64
       
        x2 = self.conv2(x1) #128
        x3 = self.conv3(x2) #256

        #########
        x4 = self.conv4(x3)  #512
        x5 = self.conv5(x4) #512
        x6 =self.conv6(x5)   #1024
        #######
        x41 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)
        x51 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=True)
        x61 = F.interpolate(x6, size=(h, w), mode='bilinear', align_corners=True)

        ########mask
        x_r =torch.cat((x41,x51,x61),1)
        x_r = self.out(x_r)
        
        #######edge
        x_e =torch.cat((x41,x51,x61),1)
        x_e =self.out(x_e)


        #######fusion###############

        x_fu =torch.cat((x_r, x_e),1)
        x_fu =self.out1(x_fu)

        # print(x_fu.shape)
        # print(x_r.shape)
        # print(x_e.shape)





       
       
 
       









        # seg = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # edge_out = F.interpolate(edge_out, size=(h, w), mode='bilinear', align_corners=True)

    
        return  x_r, x_e, x_fu

if __name__ == '__main__':
    ras = DCANet(n_channels=1, n_classes=1).cuda()
    print(ras)
    input_tensor = torch.randn(4, 1, 96, 96).cuda()

    x_r, x_e, x_fu= ras(input_tensor)


#Code by kun wang@2021.9.1
   
