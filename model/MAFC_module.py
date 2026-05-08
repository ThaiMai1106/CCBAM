import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1,pool_types=None):
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.pool_types = pool_types
        if gate_channel // reduction_ratio == 0: #fixed for mobileNetV2
            reduction_ratio = gate_channel
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(len(pool_types)*gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
        print("vao day:" + str(pool_types))
    def forward(self, x):
        # #avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        # avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # stdf = torch.std(x,(2,3),unbiased=True)#compute standard deviation
        # stdf = stdf.reshape(stdf.size()[0],stdf.size()[1],1,1)#resize to be (,1,1) the same as out put of AdaptiveAvgPool2d , i.e., self.squeeze(residual)
        # squeeze = torch.cat((stdf,avg_pool),dim=1)
        # #return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # return self.gate_c(squeeze).unsqueeze(2).unsqueeze(3).expand_as(x)
        squeeze_all = self.get_channel_features(x,self.pool_types)
        return self.gate_c(squeeze_all).unsqueeze(2).unsqueeze(3).expand_as(x)
    def get_channel_features(self,x,pool_types):
        squeeze_all = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                squeeze = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type=='max':
                squeeze = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type=='std':
                stdf = torch.std(x,(2,3),unbiased=True)#compute standard deviation
                squeeze = stdf.reshape(stdf.size()[0],stdf.size()[1],1,1)#resize to be (,1,1) the same as out put of AdaptiveAvgPool2d , i.e., self.squeeze(residual)
            if squeeze_all is None:
                squeeze_all = squeeze
            else:
                squeeze_all = torch.cat((squeeze_all,squeeze),1)
        return squeeze_all
class DeptSpatial(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=8):
        super().__init__()
        mid = gate_channel // reduction_ratio
        # self.convdw = nn.Conv2d(mid, mid, kernel_size=7, stride=1, padding=3, groups=mid, bias=False)
        # Optional: use this to change the kernel size of the depthwise convolution
        self.convdw = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=False)

        self.gate_s = nn.Sequential()
        self.gate_s.add_module( "gate_s_conv_reduce0", nn.Conv2d(gate_channel, mid, kernel_size=1))
        self.gate_s.add_module( "gate_s_conv_depthwise", self.convdw)
        self.gate_s.add_module( "gate_s_bn0", nn.BatchNorm2d(mid))
        self.gate_s.add_module( "gate_s_relu0", nn.ReLU(inplace=True))
        self.gate_s.add_module( "gate_s_conv_reduce",nn.Conv2d(mid, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor)

class BAMM(nn.Module):
    def __init__(self, gate_channel, pool_types=None):
        super(BAMM, self).__init__()
        self.channel_att = ChannelGate(gate_channel,pool_types=pool_types)
        self.spatial_att = DeptSpatial(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor
