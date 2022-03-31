import logging
import math
import os
import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from SETR.transformer_model import TransModel2d, TransConfig
from torchvision import models
from SegTrGAN.wassp import VAMM1, PyramidPooling


import math 
#########
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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
	

class GEncoder(nn.Module):
	def __init__(self, n_channels, n_classes, deep_supervision = False):
		super(GEncoder, self).__init__()
		self.deep_supervision = deep_supervision
		# self.inc = inconv(n_channels, 64)
		# self.down1 = down(64, 128)
		# self.down2 = down(128, 256)
		# self.down3 = down(256, 512)
		# self.down4 = down(512, 1024)
        

        ################################vgg16#######################################
		feats = list(models.vgg16_bn(pretrained=True).features.children())
		feats[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv1 = nn.Sequential(*feats[:6])
        # print(self.conv1)
		self.conv2 = nn.Sequential(*feats[6:13])
		self.conv3 = nn.Sequential(*feats[13:23])
		self.conv4 = nn.Sequential(*feats[23:33])
		self.conv5 = nn.Sequential(*feats[33:43])
        #######################################################################
         # ---- edge branch ----
		self.edge_conv1 = BasicConv2d(128, 64, kernel_size=1)
		self.edge_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
		self.edge_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
		self.edge_conv4 = BasicConv2d(64, n_classes, kernel_size=3, padding=1)




	def forward(self, x):
		# x1 = self.inc(x)
		# # print(x1.shape)
		# x2 = self.down1(x1)
		# x3 = self.down2(x2)
		# x4 = self.down3(x3)
		# x5 = self.down4(x4)
       
       





		x1 =self.conv1(x)
		# print(x1.shape)
		x2 =self.conv2(x1)
		# print(x2.shape)
		x3 =self.conv3(x2)
		# print(x3.shape)
		x4 =self.conv4(x3)
		# print(x4.shape)
		x5 =self.conv5(x4)
		# print(x5.shape)


        # ---- edge guidance ----
		x_e = self.edge_conv1(x2)
		# print(x_e.shape)
		x_e = self.edge_conv2(x_e)
		# print(x_e.shape)
		edge_guidance = self.edge_conv3(x_e)  # torch.Size([1, 64, 48, 48])
		edge_in = F.interpolate(edge_guidance,
                                     scale_factor=2,                       ###抽出边缘特征
                                     mode='bilinear')
		# print(edge_in.shape)
		lateral_edge = self.edge_conv4(edge_guidance)   # NOTES: Sup-2 (bs, 64, 48, 48) -> (bs, 1, 48, 48)
		# print(lateral_edge.shape)
		lateral_edge = F.interpolate(lateral_edge,
                                     scale_factor=2,
                                     mode='bilinear')
		# print(lateral_edge.shape)
		
		if self.deep_supervision:
			x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
			x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
			x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
			x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
			
			return x0, x11, x22, x33, x44
		else:
			return  x1, x2, x3, x4, x5, edge_in, lateral_edge


# class GDcoder(nn.Module):
# 	def __init__(self, n_channels, n_classes, deep_supervision = False):
# 		super(GEncoder, self).__init__()
# 		self.deep_supervision = deep_supervision
#         self.up1 = up(1024, 256)
# 		self.up2 = up(512, 128)
# 		self.up3 = up(256, 64)
# 		self.up4 = up(128, 64)
# 		self.outc = outconv(64, n_classes)
#         self.GEncoder =  GEncoder( n_channels=1, n_classes=1)   #注意自己设置 

# 	def forward(self, x):
#         x1,x2,x3,x4,x5 = self.GEncoder(x)
# 		x44 = self.up1(x5, x4)
# 		x33 = self.up2(x44, x3)
# 		x22 = self.up3(x33, x2)
# 		x11 = self.up4(x22, x1)
# 		x0 = self.outc(x11)
       
		
# 		if self.deep_supervision:
# 			x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
# 			x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
# 			x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
# 			x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
			
# 			return x0, x11, x22, x33, x44
# 		else:
# 			return  x5, x0



# class UNet(nn.Module):
# 	def __init__(self, n_channels, n_classes, deep_supervision = False):
# 		super(UNet, self).__init__()
# 		self.deep_supervision = deep_supervision
# 		self.inc = inconv(n_channels, 64)
# 		self.down1 = down(64, 128)
# 		self.down2 = down(128, 256)
# 		self.down3 = down(256, 512)
# 		self.down4 = down(512, 512)
# 		self.up1 = up(1024, 256)
# 		self.up2 = up(512, 128)
# 		self.up3 = up(256, 64)
# 		self.up4 = up(128, 64)
# 		self.outc = outconv(64, n_classes)
		
# 		self.dsoutc4 = outconv(256, n_classes)
# 		self.dsoutc3 = outconv(128, n_classes)
# 		self.dsoutc2 = outconv(64, n_classes)
# 		self.dsoutc1 = outconv(64, n_classes)

# 	def forward(self, x):
# 		x1 = self.inc(x)
# 		# print(x1.shape)
# 		x2 = self.down1(x1)
# 		# print(x2.shape)
# 		x3 = self.down2(x2)
# 		# print(x3.shape)
# 		x4 = self.down3(x3)
# 		# print(x4.shape)
# 		x5 = self.down4(x4)
# 		# print(x5.shape)
# 		x44 = self.up1(x5, x4)
# 		x33 = self.up2(x44, x3)
# 		x22 = self.up3(x33, x2)
# 		x11 = self.up4(x22, x1)
# 		x0 = self.outc(x11)
# 		if self.deep_supervision:
# 			x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
# 			x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
# 			x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
# 			x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')
			
# 			return x0, x11, x22, x33, x44
# 		else:
# 			return x5, x0




#########










class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.is_segmentation = is_segmentation
    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1 
        ww = w // p2 

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)
        
        encode_x = self.bert_model(x)[-1] # 取出来最后一层
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        return encode_x, x 


class PreTrainModel(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Vit(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        sample_rate=4,
                        ):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x) 
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
       
        self.final_out = nn.Conv2d(features[-2], out_channels, 3, padding=1)
        self.Gdcoder = GEncoder(n_channels=1, n_classes=1)
       

    def forward(self, xg, x):
        x1, x2, x3, x4, x5, edge_in,lateral_edge = self.Gdcoder(xg)
        


        x11 = self.decoder_1(x)+x4
        x12 = self.decoder_2(x11)+x3
        x13 = self.decoder_3(x12)+x2
        x14 = self.decoder_4(x13)+x1
        # print(edge_in.shape)
       
        x14 =torch.cat((edge_in, x14), dim=1)
        # print(x14.shape)
        x0 = self.final_out(x14)
        return  x0, lateral_edge

class NetS(nn.Module):
    def __init__(self, patch_size=(32, 32), 
                        in_channels=1, 
                        out_channels=1, 
                        hidden_size=1024, 
                        num_hidden_layers=8,
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 128],
                        sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
                        
                   
        self.Gdcoder = GEncoder(n_channels=1, n_classes=1)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels, features=decode_features)
        # self.cov= nn.Conv2d(1024*2, 1024,1)
        self.cov1= nn.Conv2d(1024, 512,1)
        self.VAMM1 = VAMM1(1024,dilation_level=[1,2,4,8])
        self.PyramidPooling =PyramidPooling(1024,1024)
    def forward(self, x):
        x1, x2, x3, x4, x5, edge_in,lateral_edge = self.Gdcoder(x)    
        _, final_x = self.encoder_2d(x)
        final_x =self.cov1(final_x)
        final_x =torch.cat((final_x, x5), dim=1)
          # 信息融合
        # final_x = self.cov(final_x)
        final_x = self.VAMM1(final_x)
        final_x = self.PyramidPooling(final_x)
           
        x0, lateral_edge =self.decoder_2d(x,final_x)
        return x0, lateral_edge 




# class NetC(nn.Module):
# 	def __init__(self, n_channels, n_classes):
# 		super(NetC, self).__init__()
		
# 		self.inc = inconv(n_channels, 64)
# 		self.down1 = down(64, 128)
# 		self.down2 = down(128, 256)
# 		self.down3 = down(256, 512)
# 		self.down4 = down(512, 512)
# 		self.outc = outconv(64, n_classes)
		
# 		self.dsoutc4 = outconv(256, n_classes)
# 		self.dsoutc3 = outconv(128, n_classes)
# 		self.dsoutc2 = outconv(64, n_classes)
# 		self.dsoutc1 = outconv(64, n_classes)

# 	def forward(self, x):

        
# 		x1 = self.inc(x)
# 		batchsize = x.size()[0]
# 		x2 = self.down1(x1)
	
# 		x3 = self.down2(x2)
# 		x4 = self.down3(x3)
# 		x5 = self.down4(x4)

        
# 		output = torch.cat((x.view(batchsize,-1),1*x1.view(batchsize,-1),
#                                 2*x2.view(batchsize,-1),2*x3.view(batchsize,-1),
#                                 2*x4.view(batchsize,-1),2*x5.view(batchsize,-1)),1)

       
	
# # 		return output  
# class NetC(nn.Module):
#     def __init__(self, patch_size=(32, 32), 
#                         in_channels=1, 
#                         out_channels=1, 
#                         hidden_size=1024, 
#                         num_hidden_layers=8, 
#                         num_attention_heads=16,
#                         decode_features=[512, 256, 128, 64],
#                         sample_rate=4,):
#         super().__init__()
#         config = TransConfig(patch_size=patch_size, 
#                             in_channels=in_channels, 
#                             out_channels=out_channels, 
#                             sample_rate=sample_rate,
#                             hidden_size=hidden_size, 
#                             num_hidden_layers=num_hidden_layers, 
#                             num_attention_heads=num_attention_heads)
                        
                   
#         # self.Gdcoder = GEncoder(n_channels=1, n_classes=1)
#         self.encoder_2d = Encoder2D(config)
#         # self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels, features=decode_features)
#         # self.cov= nn.Conv2d(1024*2, 1024,1)
#     def forward(self, x):
#         # x1, x2, x3, x4, x5  = self.Gdcoder(x)    
#         _, final_x = self.encoder_2d(x)
#         batchsize = x.size()[0]
#         # final_x =torch.cat((final_x, x5), dim=1)  # 信息融合
#         # final_x = self.cov(final_x)   
#         # x0 =self.decoder_2d(final_x)
#         output = final_x.view(batchsize,-1)
#         return output    

if __name__ == '__main__':
    ras =GEncoder(n_channels=1, n_classes=1).cuda()
    input_tensor = torch.randn(4, 1, 96, 96).cuda()
    out = ras(input_tensor)
    # print(out[0].shape)




#code by kun wang