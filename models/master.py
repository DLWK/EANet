import torch
from torchvision import models
import torch.nn as nn

# from resnet import resnet34
# import resnet
from torch.nn import functional as F
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x



class DSC(nn.Module):
    def __init__(self, in_channels):
        super(DSC, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)
        self.norm = nn.Sigmoid()
        self.conv1= nn.Conv2d(in_channels*2, 1, kernel_size=1, padding=0)
        self.dconv1=nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()

        branches_1=self.conv3x3(x)
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1) 

        feat_g =feat
        # print(feat_g.shape)
        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)
        
        out1 = feat_g * feat_g1
        out1 = self.dconv1(out1)
       
     





        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2 +out1
       


        feat1=torch.cat([fusion_1_2,branches_3],dim=1)

        feat_g =feat1
        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)
        out2 = feat_g * feat_g1
        out2 = self.dconv1(out2)


        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
      
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
      
        att_3=att1[:,1,:,:].unsqueeze(1)
       

        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3 +out2)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax





   











class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        # x = self.se(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x


