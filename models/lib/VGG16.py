import logging
import math
import os
import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
# from SETR.transformer_model import TransModel2d, TransConfig
from torchvision import models
from models.multi_scale_module import ASPP
import math 
#########
import torch
import torch.nn as nn
import torch.nn.functional as F
from SegTrGAN.wassp import VAMM1, PyramidPooling
import warnings
warnings.filterwarnings(action='ignore')


class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''

	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			double_conv(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=True):
		super(up, self).__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self,x1, x2):
		x1 = self.up(x1)

		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2))

		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x
	
class Backbone(nn.Module):
	def __init__(self, n_channels, n_classes, deep_supervision = False):
		super(Backbone, self).__init__()
		self.deep_supervision = deep_supervision
		
        ################################vgg16#######################################
		feats = list(models.vgg16_bn(pretrained=True).features.children())
		feats[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   #适应任务
		self.conv1 = nn.Sequential(*feats[:6])
        # print(self.conv1)
		self.conv2 = nn.Sequential(*feats[6:13])
		self.conv3 = nn.Sequential(*feats[13:23])
		self.conv4 = nn.Sequential(*feats[23:33])
		self.conv5 = nn.Sequential(*feats[33:43])
        ################################Gate#######################################
		# self.down1 = down(64, 128)
		# self.down2 = down(128, 256)
		# self.down3 = down(256, 512)
		# self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		# self.sap = VAMM1(512,dilation_level=[1,2,4,8])
		self.aspp = ASPP(512,512)
		# self.PyramidPooling =PyramidPooling(512,512)   #trick
        #   body
		# self.up1b = up(1024, 256)
		# self.up2b = up(512, 128)
		# self.up3b = up(256, 64)
		# self.up4b = up(128, 64)

		#  #  detail
		# self.up1d = up(1024, 256)
		# self.up2d = up(512, 128)
		# self.up3d = up(256, 64)
		# self.up4d = up(128, 64)
		
		# self.outc = outconv(64*3, n_classes)
		self.outc = outconv(64, n_classes)


		# self.dsc = DSConv3x3(64*2, 64)
		# self.dsoutc4 = outconv(256, n_classes)
		# self.dsoutc3 = outconv(128, n_classes)
		# self.dsoutc2 = outconv(64, n_classes)
		# self.dsoutc1 = outconv(64, n_classes)

	def forward(self, x):
		# x1 = self.inc(x)
	
		# x2 = self.down1(x1)
		# x3 = self.down2(x2)
		# x4 = self.down3(x3)
		# x5 = self.down4(x4)
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)
		x5 =self.aspp(x5)
		# x5=self.sap(x5)
		# x5 =self.PyramidPooling(x5)
        #Mask
		x44 = self.up1(x5, x4)
		x33 = self.up2(x44, x3)
		x22 = self.up3(x33, x2)
		x11 = self.up4(x22, x1)
        # #Body
		# x44b = self.up1b(x5, x4)
		# x33b = self.up2b(x44b, x3)
		# x22b = self.up3b(x33b, x2)
		# x11b = self.up4b(x22b, x1)

        # #Detail
		# x44d = self.up1d(x5, x4)
		# x33d = self.up2d(x44d, x3)
		# x22d = self.up3d(x33d, x2)
		# x11d = self.up4d(x22d, x1)

		
		# # x0 = self.outs(x11)
		# xb=  self.outs(x11b)
		# xd= self.outs(x11d)

		# xf = torch.cat((x11b, x11d), dim=1)  #fusion

		# xf = self.dsc(xf)
		# xf = self.outs(xf)
		# # print(xf.shape)
		
	

		x0 = self.outc(x11)

		if self.deep_supervision:
			x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
			x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
			x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
			x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
			
			return x0, x11, x22, x33, x44
		else:
			return x0


# if __name__ == '__main__':
#     ras =Backbone(n_channels=1, n_classes=1).cuda()
#     input_tensor = torch.randn(4, 1, 96, 96).cuda()
#     out = ras(input_tensor)
#     print(out[0].shape)





#code by kun wang