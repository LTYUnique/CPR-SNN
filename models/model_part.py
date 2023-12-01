import torch
import torch.nn as nn
from prompt import Prompt

# UNet的一大层，包含了两层小的卷积
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlocks(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlocks, self).__init__()
        self.left_conv = DoubleConv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2, 2)

    def forward(self, x):
        # print('down', x.shape)
        x = self.left_conv(x)
        x_down = self.down(x)
        return x, x_down


class UpBlocks(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlocks, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.right_conv = DoubleConv(in_ch, out_ch)

        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.s = s

    def forward(self, x, x_left):
        # print(self.in_ch, '-to-', self.out_ch)
        # exit()
        x = self.up(x)
        # print( x.shape, x_left.shape)
        # exit()
        x = torch.cat((x, x_left), dim=1)
        x = self.right_conv(x)
        return x
       

class UpBlocks2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlocks2, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.right_conv = DoubleConv(in_ch, out_ch)

        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.s = s

    def forward(self, x, x_left):
        # print(self.in_ch, '-to-', self.out_ch)
        # exit()
        x = self.right_conv(x) # 通道倍减
        # print( x.shape, x_left.shape)
        # exit()
        x = torch.cat((x, x_left), dim=1)
        x = self.up(x)
        return x

class UpBlocks3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlocks3, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.right_conv1 = DoubleConv(in_ch, out_ch)
        self.right_conv2 = DoubleConv(in_ch, out_ch)

        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.s = s

    def forward(self, x, x_left, x_left_prompt):
        # print(self.in_ch, '-to-', self.out_ch)
        # exit()
        x = self.right_conv1(x) # 通道倍减
        x = torch.cat((x, x_left_prompt), dim=1)
        x = self.up(x)
        x = torch.cat((x, x_left), dim=1)
        x = self.right_conv2(x) # 通道倍减


        return x



class self_attention(nn.Module):
    r"""
        Create global dependence.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_channles):
        super(self_attention, self).__init__()
        self.in_channels = in_channles

        self.f = nn.Conv3d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.g = nn.Conv3d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.h1 = nn.Conv3d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.h2 = nn.Conv3d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.softmax_ = nn.Softmax(dim=2)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.init_weight(self.f)
        self.init_weight(self.g)
        self.init_weight(self.h1)
        self.init_weight(self.h2)

    def init_weight(self, conv):
        nn.init.kaiming_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()
    '''
    def shared_attention(self, x1, x2, batch_size, channels, height, width):
        # print(x1.size())
        # exit()
        # torch.Size([1, 8, 32, 64, 64])
        # torch.Size([1, 8, 64, 64])
        # 看看2D的什么输出

        # x1
        # k, q
        f = self.f(x1).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x1).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)
        # v
        h = self.h1(x1).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W
        x1 = self.gamma1 * self_attention_map + x1

        # x2
        # k, q
        f = self.f(x2).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x2).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)
        # v
        h = self.h2(x2).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W
        x2 = self.gamma2 * self_attention_map + x2

        return x1, x2
    '''
    def forward(self, x1, x2):
        batch_size, channels, z, height, width = x1.size()
        # print(x1.size())
        '''
        torch.Size([1, 8, 30, 64, 64])
        torch.Size([1, 16, 15, 32, 32])
        torch.Size([1, 32, 7, 16, 16])
        '''
        '''
        for i in range(z):
            x1[:,:,i,:,:], x2[:,:,i,:,:] = self.shared_attention(x1[:,:,i,:,:], x2[:,:,i,:,:], batch_size, channels, height, width)
        '''
        # x1
        # k, q
        f = self.f(x1).view(batch_size, -1, height * width*z).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x1).view(batch_size, -1, height * width*z)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)
        # v
        h = self.h1(x1).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, z,  height, width)  # B * C * H * W
        x1 = self.gamma1 * self_attention_map + x1

        # x2
        # k, q
        f = self.f(x2).view(batch_size, -1, height * width*z).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x2).view(batch_size, -1, height * width*z)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)
        # v
        h = self.h2(x2).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, z, height, width)  # B * C * H * W
        x2 = self.gamma2 * self_attention_map + x2
        return x1, x2


class empty(nn.Module):
    def forward(self, x1, x2):
        return x1
    
class EPA_attn(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"

        这里取消了cat操作，将两个特征图分别返回，作为新的x1, x2
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1, prompt=False, n_clin_var=None, batch_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        if prompt:
            assert batch_size and n_clin_var, 'batch size should be given if want prompt'
            self.prompt = Prompt(batch_size, hidden_size, input_size, n_clin_var)
        else:
            self.prompt = empty()

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

        self.norm = nn.LayerNorm(hidden_size//2) # TODO: norm!!!
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(hidden_size//2), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(hidden_size//2), requires_grad=True)
    
    def forward(self, x1, x2, clin_var):
        B_, C_, H_, W_, D_ = x1.shape  # 2 16 64 64 32
        x1 = x1.reshape(B_, C_, H_ * W_ * D_).permute(0, 2, 1)
        x1=self.norm(x1)
        x2 = x2.reshape(B_, C_, H_ * W_ * D_).permute(0, 2, 1)
        x2=self.norm(x2)  # todo: x2==self.norm(x2)

        x = torch.cat((x1, x2), dim=-1)
        # print('before', x.shape, clin_var.shape, '*********') # b, n c
        x = self.prompt(x, clin_var) # b, 1, 
        # print(x)
        # print('-------------', type(x), type(x[0]))
        # x = x[0]
        # print('-------------', x.shape)
        # print('-----------------------------------------------------------------------------')

        B, N, C = x.shape
        

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        # x = torch.cat((x_SA, x_CA), dim=-1)
        
        attn1 = x1 + self.gamma1 * x_SA
        attn2 = x2 + self.gamma1 * x_CA
        attn_skip1 = attn1.reshape(B_, H_, W_, D_, C_).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)  
        attn_skip2 = attn2.reshape(B_, H_, W_, D_, C_).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)  
        return attn_skip1, attn_skip2


class EPA_attn2(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"

        这里沿袭cat操作，x1, x2使用相同值。注意hidden_size（输出通道数）不能想上一个epa那样倍增了
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1, prompt=False, n_clin_var=None, batch_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        if prompt:
            assert batch_size and n_clin_var, 'batch size should be given if want prompt'
            self.prompt = Prompt(batch_size, hidden_size, input_size, n_clin_var)
        else:
            self.prompt = empty()

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 4))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 4))

        self.norm1 = nn.LayerNorm(hidden_size//2)
        self.norm2 = nn.LayerNorm(hidden_size//2)
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(hidden_size//2), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(hidden_size//2), requires_grad=True)
    
    def forward(self, x1, x2, clin_var):
        B_, C_, H_, W_, D_ = x1.shape  # 2 16 64 64 32
        x1 = x1.reshape(B_, C_, H_ * W_ * D_).permute(0, 2, 1)
        x1=self.norm1(x1)
        x2 = x2.reshape(B_, C_, H_ * W_ * D_).permute(0, 2, 1)
        x2=self.norm2(x2)

        x = torch.cat((x1, x2), dim=-1)
        # print('before', x.shape, clin_var.shape, '*********') # b, n c
        x = self.prompt(x, clin_var) # will be empty if prompt is false
        # print(x)
        # print('-------------', type(x), type(x[0]))
        # x = x[0]
        # print('-------------', x.shape)
        # print('-----------------------------------------------------------------------------')

        B, N, C = x.shape
        

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        
        attn1 = x1 + self.gamma1 * x
        attn2 = x2 + self.gamma1 * x
        attn_skip1 = attn1.reshape(B_, H_, W_, D_, C_).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)  
        attn_skip2 = attn2.reshape(B_, H_, W_, D_, C_).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D) 
        return attn_skip1, attn_skip2
