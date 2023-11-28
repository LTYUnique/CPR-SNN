import torch
import torch.nn as nn
import torch.nn.functional as F

from model_part import *
from prompt import Prompts, Prompts2, Prompts3

import torch
import torch.nn as nn


class Model2(nn.Module):
    def __init__(self, basic_channel, num_classes, x1_scale, x2_scale):
        super(Model2, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8

        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        
        self.attention1 = self_attention(basic_channel)
        self.attention2 = self_attention(out_channels[1])
        self.attention3 = self_attention(out_channels[2])

        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
    
    @torch.no_grad()
    def predict(self, x1, x2):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        # x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
    
    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        # x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model3(nn.Module):
    def __init__(self, basic_channel, num_classes, shape, x1_scale, x2_scale, n_clin_var):
        super(Model3, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn(H*W*D, basic_channel*2, H*4)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn(H*W*D, out_channels[1]*2, H*4)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn(H*W*D, out_channels[2]*2, H*4)

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
    
    @torch.no_grad()
    def predict(self, x1, x2):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
    
    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model4(nn.Module):
    '''add prompt'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model4, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn(H*W*D, basic_channel*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt1 = Prompts(H*W*D, basic_channel, 32, batch_size, 8)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn(H*W*D, out_channels[1]*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt2 = Prompts(H*W*D, out_channels[1], 32, batch_size, 8)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn(H*W*D, out_channels[2]*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt3 = Prompts(H*W*D, out_channels[2], 16, batch_size, 4)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        # x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        # x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        # x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model5(nn.Module):
    '''epa使用另一种方式。x1, x2接收同样的epa输出。 prompt忘改了，现在还是加在了epa的中间'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model5, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt1 = Prompts(H*W*D, basic_channel, 32, batch_size, 8)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt2 = Prompts(H*W*D, out_channels[1], 32, batch_size, 8)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4, prompt=True, n_clin_var=n_clin_var, batch_size=batch_size)
        # self.prompt3 = Prompts(H*W*D, out_channels[2], 16, batch_size, 4)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        # x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        # x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        # x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model6(nn.Module):
    '''epa使用另一种方式。prompt加在epa之后'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model6, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model7(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model7, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape
        # input: b c d h w

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def save_gradient(self, grad):
        self.gradients = grad
    def cam_pcr(self, x1, x2, clin_var, layer='bottleneck'):
        act = None
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        if layer=='x1_down1':
            x1_down_1.register_hook(self.save_gradient)
            act = x1_down_1
        elif layer=='x2_down1':
            x2_down_1.register_hook(self.save_gradient)
            act = x2_down_1
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        if layer=='x1_down1_attention':
            x1_down_1.register_hook(self.save_gradient)
            act = x1_down_1
        elif layer=='x2_down1_attention':
            x2_down_1.register_hook(self.save_gradient)
            act = x2_down_1
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)
        if layer=='x1_down1_prompt':
            x1_down_1.register_hook(self.save_gradient)
            act = x1_down_1
        elif layer=='x2_down1_prompt':
            x2_down_1.register_hook(self.save_gradient)
            act = x2_down_1
        
        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        if layer=='x1_down2':
            x1_down_2.register_hook(self.save_gradient)
            act = x1_down_2
        elif layer=='x2_down2':
            x2_down_2.register_hook(self.save_gradient)
            act = x2_down_2
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        if layer=='x1_down2_attention':
            x1_down_2.register_hook(self.save_gradient)
            act = x1_down_2
        elif layer=='x2_down2_attention':
            x2_down_2.register_hook(self.save_gradient)
            act = x2_down_2
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)
        if layer=='x1_down2_prompt':
            x1_down_2.register_hook(self.save_gradient)
            act = x1_down_2
        elif layer=='x2_down2_prompt':
            x2_down_2.register_hook(self.save_gradient)
            act = x2_down_2

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        if layer=='x1_down3':
            x1_down_3.register_hook(self.save_gradient)
            act = x1_down_3
        elif layer=='x2_down3':
            x2_down_3.register_hook(self.save_gradient)
            act = x2_down_3
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        if layer=='x1_down3_attention':
            x1_down_3.register_hook(self.save_gradient)
            act = x1_down_3
        elif layer=='x2_down3_attention':
            x2_down_3.register_hook(self.save_gradient)
            act = x2_down_3
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)
        if layer=='x1_down3_prompt':
            x1_down_3.register_hook(self.save_gradient)
            act = x1_down_3
        elif layer=='x2_down3_prompt':
            x2_down_3.register_hook(self.save_gradient)
            act = x2_down_3

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        if layer=='x1_down4':
            x1_down_4.register_hook(self.save_gradient)
            act = x1_down_4
        elif layer=='x2_down4':
            x2_down_4.register_hook(self.save_gradient)
            act = x2_down_4
        bn = self.bottleneck(x1_down_4 + x2_down_4)
        if layer=='bottleneck':
            bn.register_hook(self.save_gradient)
            act = bn

        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, act

    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model8(nn.Module):
    '''两次跳层连接两次cat，prompt之前一次，prompt之后一次'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model8, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks3(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks3(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks3(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

class Model9(nn.Module):
    '''Model6的基础上改的，也就是现在的模型去掉分类和prompt，看看分割的结果能不能回去'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Model9, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        # self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)

        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)

        # # TODO: 3D!!
        # out = F.avg_pool3d(bn, bn.size()[-3:])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        return self.OutConv(up_1)
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        x1_left_1, x1_down_1 = self.x1_down_block1(x1)
        x2_left_1, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)

        x1_left_2, x1_down_2 = self.x1_down_block2(x1_down_1)
        x2_left_2, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_left_3, x1_down_3 = self.x1_down_block3(x1_down_2)
        x2_left_3, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_left_3*self.x1_scale+x2_left_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_left_2*self.x1_scale+x2_left_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_left_1*self.x1_scale+x2_left_1*self.x2_scale)


        # out = F.avg_pool3d(bn, bn.size()[-3:])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1) , None


'''
再model7的基础上做消融实验

'''
## imput image
# exp1-1 use T1 only

class Ablation1_1_T1(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation1_1_T1, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x1)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x1)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

# exp1-2 use T2 only

class Ablation1_2_T2(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation1_2_T2, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x2)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x2)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

## feature merge
# exp2-1 change epa to add
class Ablation2_1_add(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation2_1_add, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        # self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        # self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        # self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = x1_down_1 + x2_down_1, x1_down_1 + x2_down_1
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = x1_down_2 + x2_down_2, x1_down_2 + x2_down_2
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = x1_down_3 + x2_down_3, x1_down_3 + x2_down_3
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = x1_down_1 + x2_down_1, x1_down_1 + x2_down_1
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = x1_down_2 + x2_down_2, x1_down_2 + x2_down_2
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = x1_down_3 + x2_down_3, x1_down_3 + x2_down_3
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

# exp2-2 change epa to cat
class Ablation2_2_cat(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation2_2_cat, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = nn.Sequential(
            nn.Conv3d(basic_channel*2, basic_channel, 3, padding=1),
            nn.BatchNorm3d(basic_channel),
            nn.ReLU(inplace=True)
        )
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = nn.Sequential(
            nn.Conv3d(out_channels[1]*2, out_channels[1], 3, padding=1),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = nn.Sequential(
            nn.Conv3d(out_channels[2]*2, out_channels[2], 3, padding=1),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True)
        )
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        down_1 = self.attention1(torch.cat((x1_down_1, x2_down_1), dim=1))
        x1_down_1, x2_down_1 = self.prompt1(down_1, down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        down_2 = self.attention2(torch.cat((x1_down_2, x2_down_2), dim=1))
        x1_down_2, x2_down_2 = self.prompt2(down_2, down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        down_3 = self.attention3(torch.cat((x1_down_3, x2_down_3), dim=1))
        x1_down_3, x2_down_3 = self.prompt3(down_3, down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        down_1 = self.attention1(torch.cat((x1_down_1, x2_down_1), dim=1))
        x1_down_1, x2_down_1 = self.prompt1(down_1, down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        down_2 = self.attention2(torch.cat((x1_down_2, x2_down_2), dim=1))
        x1_down_2, x2_down_2 = self.prompt2(down_2, down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        down_3 = self.attention3(torch.cat((x1_down_3, x2_down_3), dim=1))
        x1_down_3, x2_down_3 = self.prompt3(down_3, down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

## loss func
# exp3-1 no seg loss

class Ablation3_1_noSeg(nn.Module):  # 需要改loss function
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation3_1_noSeg, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        
        self.linear = nn.Linear(out_channels[-1], num_classes)

        

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return None, None, None, None, out

# exp3-2 different loss conf

## prompt
# exp4-1 no prompt

class Ablation4_1_noPrompt(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation4_1_noPrompt, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        # self.prompt1 = Prompts(batch_size, basic_channel, H*W*D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        # self.prompt2 = Prompts(batch_size, out_channels[1], H*W*D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        # self.prompt3 = Prompts(batch_size, out_channels[2], H*W*D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, _):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, None)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, None)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, None)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, None)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, None)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, None)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

# exp4-2 no cross attention

class Ablation4_2_noCA(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation4_2_noCA, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts2(basic_channel, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts2(out_channels[1], n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts2(out_channels[2], n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out

# exp4-3 no cross attention cat

class Ablation4_2_noCA_cat(nn.Module):
    '''把prompt之后的特征图做跳层连接，上采样块先cat再上采样'''
    def __init__(self, basic_channel, num_classes, shape, batch_size, x1_scale, x2_scale, n_clin_var):
        super(Ablation4_2_noCA_cat, self).__init__()
        self.x1_scale = x1_scale
        self.x2_scale = x2_scale
        in_ch = 1
        rate = [1,2,4,8,16]
        out_channels = [basic_channel*i for i in rate]
        H, W, D = shape

        self.x1_down_block1 = DownBlocks(in_ch, basic_channel)  # 8, 30, 64, 64 
        self.x2_down_block1 = DownBlocks(in_ch, basic_channel)
        H, W, D = H//2, W//2, D//2
        self.attention1 = EPA_attn2(H*W*D, basic_channel*2, H*4)
        self.prompt1 = Prompts3(batch_size, basic_channel, H, W, D, n_clin_var)
        
        self.x1_down_block2 = DownBlocks(basic_channel, out_channels[1])  # 16, 15, 32, 32
        self.x2_down_block2 = DownBlocks(basic_channel, out_channels[1])
        H, W, D = H//2, W//2, D//2
        self.attention2 = EPA_attn2(H*W*D, out_channels[1]*2, H*4)
        self.prompt2 = Prompts3(batch_size, out_channels[1], H, W, D, n_clin_var)
        
        self.x1_down_block3 = DownBlocks(out_channels[1], out_channels[2])  # 32, 7, 16, 16
        self.x2_down_block3 = DownBlocks(out_channels[1], out_channels[2])
        H, W, D = H//2, W//2, D//2
        self.attention3 = EPA_attn2(H*W*D, out_channels[2]*2, H*4)
        self.prompt3 = Prompts3(batch_size, out_channels[2], H, W, D, n_clin_var)
        # print('prompt3')

        self.x1_down_block4 = DownBlocks(out_channels[2], out_channels[3])  # 64, 3, 8, 8
        self.x2_down_block4 = DownBlocks(out_channels[2], out_channels[3])
        self.bottleneck = DoubleConv(out_channels[-2], out_channels[-1])  # 128, 3, 8, 8
        # print('bottleneck')

        self.up_block4 = UpBlocks(out_channels[-1], out_channels[-2])  # 3 -> 6
        self.up_block3 = UpBlocks2(out_channels[-2], out_channels[-3])  # 6 -> 12
        self.up_block2 = UpBlocks2(out_channels[-3], out_channels[-4])  # 12 -> 24
        self.up_block1 = UpBlocks2(out_channels[-4], basic_channel)  # 24 -> 48
        # print('up_block1')

        self.linear = nn.Linear(out_channels[-1], num_classes)

        self.Up4Conv = nn.Sequential(
            nn.Conv3d(out_channels[-2], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up3Conv = nn.Sequential(
            nn.Conv3d(out_channels[-3], in_ch, 1),
            nn.Sigmoid()
        )
        self.Up2Conv = nn.Sequential(
            nn.Conv3d(out_channels[-4], in_ch, 1),
            nn.Sigmoid()
        )
        self.OutConv = nn.Sequential(
            nn.Conv3d(basic_channel, in_ch, 1),
            nn.Sigmoid()  # BCELoss
        )
        # print('OutConv')

    @torch.no_grad()
    def predict(self, x1, x2, clin_var):
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.OutConv(up_1), out
 
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape)
        # exit()
        _, x1_down_1 = self.x1_down_block1(x1)
        _, x2_down_1 = self.x2_down_block1(x2)
        x1_down_1, x2_down_1 = self.attention1(x1_down_1, x2_down_1, clin_var)
        x1_down_1, x2_down_1 = self.prompt1(x1_down_1, x2_down_1, clin_var)

        _, x1_down_2 = self.x1_down_block2(x1_down_1)
        _, x2_down_2 = self.x2_down_block2(x2_down_1)
        x1_down_2, x2_down_2 = self.attention2(x1_down_2, x2_down_2, clin_var)
        x1_down_2, x2_down_2 = self.prompt2(x1_down_2, x2_down_2, clin_var)

        _, x1_down_3 = self.x1_down_block3(x1_down_2)
        _, x2_down_3 = self.x2_down_block3(x2_down_2)
        x1_down_3, x2_down_3 = self.attention3(x1_down_3, x2_down_3, clin_var)
        x1_down_3, x2_down_3 = self.prompt3(x1_down_3, x2_down_3, clin_var)

        x1_left_4, x1_down_4 = self.x1_down_block4(x1_down_3)
        x2_left_4, x2_down_4 = self.x2_down_block4(x2_down_3)
        # x1_down_4 = self.attention4(x1_down_4)
        # x2_down_4 = self.attention4(x2_down_4)

        bn = self.bottleneck(x1_down_4 + x2_down_4)

        up_4 = self.up_block4(bn, x1_left_4*self.x1_scale+x2_left_4*self.x2_scale)
        up_3 = self.up_block3(up_4, x1_down_3*self.x1_scale+x2_down_3*self.x2_scale)
        up_2 = self.up_block2(up_3, x1_down_2*self.x1_scale+x2_down_2*self.x2_scale)
        up_1 = self.up_block1(up_2, x1_down_1*self.x1_scale+x2_down_1*self.x2_scale)

        # TODO: 3D!!
        out = F.avg_pool3d(bn, bn.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return self.Up4Conv(up_4), self.Up3Conv(up_3), self.Up2Conv(up_2), self.OutConv(up_1), out






if __name__ == '__main__':
     model = Model3(16, 2, (32,32,16),1,1)
     model.cuda()
     T1 = torch.randn(2, 1, 32, 32, 16).cuda()
     T2 = torch.randn(2, 1, 32, 32, 16).cuda()
     y = model(T1, T2)
     print(y[-1].shape)

    # model = EPA_attn(64*64*32, 16, 64*4).cuda()
    # T1 = torch.randn(2, 8, 64, 64, 32).cuda()  # B, C, H, W, D
    # T2 = torch.randn(2, 8, 64, 64, 32).cuda()  # B, C, H, W, D
    # y1, y2 = model(T1, T2)
    # print(y1.shape, y2.shape)