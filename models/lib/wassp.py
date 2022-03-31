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



class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
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
        print(feat_g.shape)
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
        print(fusion_1_2.shape)


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
        print(att1)
        print(att1[:,0,:,:])
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
        print(att_1_2.shape)
        att_3=att1[:,1,:,:].unsqueeze(1)
        print(att_3.shape)

        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3 +out2)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax

############################
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
######################################## 




class VAMM1(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(VAMM1, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
                ])
        self.DSConv3x3 =DSConv3x3(channel, channel, stride=1, dilation=1)
        # self.branches_2 = DSConv3x3(channel, channel, stride=1, dilation=1)
        self.conv3x3=nn.Conv2d(channel, channel,dilation=1,kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channel)
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
                convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
                nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
                )

    def forward(self, x):
        x1 = self.conv(x)
        # print(self.DSConv3x3)
        # m1= F.conv2d(x1,self.conv.weight,padding=2,dilation=2)
        branch1= self.conv3x3(x1)
        branch1 =self.bn(branch1)
        branch2= F.conv2d(x1,self.conv3x3.weight,padding=2,dilation=2)
        branch2 =self.bn(branch2)
        branch3= F.conv2d(x1,self.conv3x3.weight,padding=4,dilation=4)
        branch3 =self.bn(branch3)
        branch4= F.conv2d(x1,self.conv3x3.weight,padding=8,dilation=8)
        branch4 =self.bn(branch4)
        # brs = [branch(conv) for branch in self.branches]
        # brs.append(branch1)
        # brs.append(branch2)
        # brs.append(branch3)
        # brs.append(branch4)
        brs =[branch1, branch2, branch3, branch4]
        brs.append(x1)
        gather = sum(brs)
        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)
        # print(len(self.dilation_level) + 1)
        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)
      
        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	+ x

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x








if __name__=='__main__':
   x=torch.randn(4,512,96,96)
   net = VAMM1(512,dilation_level=[1,2,4,8])
   net2= PyramidPooling(512,512)
   ax = net(x)
   ax= net2(ax)
   print(ax.shape)



   
