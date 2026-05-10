import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)



class DeptSpatial(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=8):
        super().__init__()
        mid = gate_channel // reduction_ratio
        self.convdw = nn.Conv2d(mid, mid, kernel_size=7, stride=1, padding=3, groups=mid, bias=False)
        # Optional: use this to change the kernel size of the depthwise convolution
        # self.convdw = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=False)
        
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( "gate_s_conv_reduce0", nn.Conv2d(gate_channel, mid, kernel_size=1))
        self.gate_s.add_module( "gate_s_conv_depthwise", self.convdw)
        self.gate_s.add_module( "gate_s_bn0", nn.BatchNorm2d(mid))
        self.gate_s.add_module( "gate_s_relu0", nn.ReLU(inplace=True))
        self.gate_s.add_module( "gate_s_conv_reduce",nn.Conv2d(mid, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor)

class DeptBAM(nn.Module):
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = DeptSpatial(gate_channel)

    def forward(self, x):
        logit = self.channel_att(x) * self.spatial_att(x)   
        att = 1 + torch.sigmoid(logit)
        return att * x
