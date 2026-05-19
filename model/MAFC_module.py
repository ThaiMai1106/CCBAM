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



class SpatialGate_LKA_Lite(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super().__init__()
        mid = max(1, gate_channel // reduction_ratio)
        self.gate_s = nn.Sequential(
            nn.Conv2d(gate_channel, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, kernel_size=7, padding=3, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, kernel_size=7, padding=9, dilation=3,
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.gate_s(x)

class DeptBAM(nn.Module):
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate_LKA_Lite(gate_channel)

    def forward(self, x):
        logit = self.channel_att(x) * self.spatial_att(x)   
        att = 1 + torch.sigmoid(logit)
        return att * x
