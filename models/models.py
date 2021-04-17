import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from lib.nn import SynchronizedBatchNorm2d
from smoothgrad import generate_smooth_grad
from guided_backprop import GuidedBackprop
from vanilla_backprop import VanillaBackprop
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
import math
from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d
from .dsc import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class EANet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False,num_filters=64, dropout=False, rate=0.1, bn=False):
        super(EANet, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg_features[0]=nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
        # ##########
        # self.vgg_features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # #######
        # self.inc1= self.vgg_features[:4]
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:18]
        self.down3 = self.vgg_features[18:27]
        self.down4 = self.vgg_features[27:36]

        self.wassp=SAPP(512)      #lastest ----  DSC module
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64+64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256+256, n_classes)
        self.dsoutc3 = outconv(128+128, n_classes)
        self.dsoutc2 = outconv(64+64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)
        self.dsoutc5 = outconv(512+512, n_classes)

        # self.fuout =outconv(5, n_classes)
        
        
        #boundray stream

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)   
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        self.expand1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1),
                                    Norm2d(64),
                                    nn.ReLU(inplace=True))
        self.expand2 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=1),
                                    Norm2d(128),
                                    nn.ReLU(inplace=True)) 
        self.expand3 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1),
                                    Norm2d(256),
                                    nn.ReLU(inplace=True))
        self.expand4 = nn.Sequential(nn.Conv2d(1, 512, kernel_size=1),
                                    Norm2d(512),
                                    nn.ReLU(inplace=True))                                                                                   


        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x55=self.wassp(x5)
       
  
      ### edge stream ####### -----  EAP module

        ss = F.interpolate(self.d0(x1), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(x2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss1 = self.gate1(ss, c3)
        # print("***********")
        # print(ss1.shape)
        ss = self.res2(ss1)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(x3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss2 = self.gate2(ss, c4)
        ss = self.res3(ss2)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(x4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss3 = self.gate3(ss, c5)
        ss = self.fuse(ss3)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)
        

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1 , x_size[2], x_size[3]))     
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()   #注意.cuda()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
       
        edge = self.expand(acts)

        edge1=self.expand1(acts)
        edge2=self.expand2(acts)
        edge3=self.expand3(acts)
        edge4=self.expand4(acts)
        
        x44 = self.up1(x55, x4)
        x33 = self.up2(x44, x3)      
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
       
        x11_con = torch.cat([x11, edge], dim=1)  #fusion
        # print(x11_con.shape)
        x0 = self.outc(x11_con)

# closed-loop -----MPR module
         #branch2
        edge1 = F.interpolate(edge1, scale_factor=1/2, mode='bilinear')
        crop_1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear')
        x_s= -1*(torch.sigmoid(crop_1)) + 1  #反转
        x22= x_s.expand(-1, 64, -1, -1).mul(x22)
        x_cat1=torch.cat([x22, edge1], dim=1)
        x_11=self.dsoutc2(x_cat1)
        x_11 = x_11+crop_1
    
        x_14 = F.interpolate(x_11, scale_factor=2, mode='bilinear')
        edge2 = F.interpolate(edge2, scale_factor=1/4, mode='bilinear')
      
   
        crop_2 = F.interpolate(x_11, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_2)) + 1  #反转
        x33= x.expand(-1, 128, -1, -1).mul(x33)
        x_cat2=torch.cat([x33, edge2], dim=1)
        x_22=self.dsoutc3(x_cat2)
        x_12 = x_22 +crop_2
        x_25 = F.interpolate(x_12, scale_factor=4, mode='bilinear')
       
        #branch4
        edge3 = F.interpolate(edge3, scale_factor=1/8, mode='bilinear')
        crop_3 = F.interpolate(x_12, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_3)) + 1  #反转
        x44= x.expand(-1, 256, -1, -1).mul(x44)
        x_cat3=torch.cat([x44, edge3], dim=1)
        x_33=self.dsoutc4(x_cat3)
        x_13 = x_33+crop_3
        x_36 = F.interpolate(x_13, scale_factor=8, mode='bilinear')
    
        #branch5
        edge4 = F.interpolate(edge4, scale_factor=1/16, mode='bilinear')
        crop_4 = F.interpolate(x_13, scale_factor=0.5, mode='bilinear')
        x= -1*(torch.sigmoid(crop_4)) + 1  #反转
        x55= x.expand(-1, 512, -1, -1).mul(x55)   #x55
        x_cat3=torch.cat([x55, edge4], dim=1)  
        x_44=self.dsoutc5(x_cat3)
        x_15 = x_44 + crop_4
        x_47 = F.interpolate(x_15, scale_factor=16, mode='bilinear')
        # x_fu = torch.cat([x0, x_14, x_25, x_36, x_47], dim=1)
        # x_fu = self.fuout(x_fu)

        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0, x_14, x_25, x_36, x_47, ss
#code by kun wang@2021.4.10