import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import warnings
import math, copy
import torchvision
warnings.filterwarnings("ignore")

from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead


class TSRN_PartialConv(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=False, hidden_units=32):
        super(TSRN_PartialConv, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
            # in_planes = 3
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            # PartialConvLayer(3, 2*hidden_units, kernel_size=9, padding=4, activation='prelu'),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    # PartialConvLayer(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1, activation='leaky_relu'),
                    # nn.BatchNorm2d(2*hidden_units)
                    BatchNorm2d(2*hidden_units)
                ))
        
        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        # block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        block_.append(PartialConvLayer(2*hidden_units, in_planes, kernel_size=9, padding=4, activation=None))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # print("x[0]", x[0].size())
        # print("x[1]", x[1].size())
        
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        
        # ----------- init input_imgs and masks ---------------
        masks = x[:, 3:, :, :]
        masks = masks.repeat(1, 3, 1, 1)
        input = x[:, :3, :, :]
        torchvision.utils.save_image(masks, "masks.png", padding=0)
        # ----------- start block1 ----------------
        block = {'1': self.block1([input, masks])}        
        # block = {'1': self.block1(x)}    
        
        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            (block['1'])
        output = torch.tanh(block[str(self.srb_nums + 3)][0])
        
        torchvision.utils.save_image(output[:, :3, :, :], "outputs.png", padding=0)
        # torchvision.utils.save_image(masks, "masks.png", padding=0)
        # output = torch.cat([output, masks[:, :1, :, :]], dim=1)
        return output


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = PartialConvLayer(channels, channels, kernel_size=3, padding=1, activation=None)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = PartialConvLayer(channels, channels, kernel_size=3, padding=1, activation=None)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        # x include imgs, masks
        outs, masks = self.conv1(x)
        outs = self.bn1(outs)
        outs = self.prelu(outs)
        masks = self.bn1(masks)
        masks = self.prelu(masks)
        
        outs, masks = self.conv2([outs, masks])
        outs = self.bn2(outs)
        masks = self.bn2(masks)
        
        outs, masks = self.gru1([outs.transpose(-1, -2).contiguous(), masks.transpose(-1, -2).contiguous()])
        outs, masks = outs.transpose(-1, -2).contiguous(), masks.transpose(-1, -2).contiguous()

        outs, masks = self.gru2([x[0] + outs, x[1] + masks])
        return outs.contiguous(), masks.contiguous()


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.conv = PartialConvLayer(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1, activation=None)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x, masks = self.conv(x)
        x = self.pixel_shuffle(x)
        masks = self.pixel_shuffle(masks)
        x = self.prelu(x)
        masks = self.prelu(masks)
        return x, masks


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = PartialConvLayer(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x include imgs, masks
        outs, masks = self.conv1(x)
        outs = outs.permute(0, 2, 3, 1).contiguous() # b, w, h, c
        masks = masks.permute(0, 2, 3, 1).contiguous() # b, w, h, c
        
        b = outs.size()
        outs = outs.view(b[0] * b[1], b[2], b[3]) # b*w, h, c
        masks = masks.view(b[0] * b[1], b[2], b[3]) # b*w, h, c
        outs, _ = self.gru(outs)
        masks, _ = self.gru(masks)
        
        # x = self.gru(x)[0]
        outs = outs.view(b[0], b[1], b[2], b[3])
        masks = masks.view(b[0], b[1], b[2], b[3])
        outs = outs.permute(0, 3, 1, 2).contiguous()
        masks = masks.permute(0, 3, 1, 2).contiguous()
        return outs, masks


class PartialConvLayer (nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bn=True, bias=False, activation="relu"):
        super().__init__()
        self.bn = bn

        # if sample == "down-7":
        # 	# Kernel Size = 7, Stride = 2, Padding = 3
        # 	self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
        # 	self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)

        # elif sample == "down-5":
        # 	self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
        # 	self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)

        # elif sample == "down-3":
        # 	self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
        # 	self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

        # else:
        # 	self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        # 	self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)

        # "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        # negative slope of leaky_relu set to 0, same as relu
        # "fan_in" preserved variance from forward pass
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        if bn:
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            # Applying BatchNorm2d layer after Conv will remove the channel mean
            self.batch_normalization = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            # Used between all encoding layers
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            # Used between all decoding layers (Leaky RELU with alpha = 0.2)
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()

    def forward(self, input):
        input_x, mask = input[0], input[1]
        # output = W^T dot (X .* M) + b
        output = self.input_conv(input_x * mask)

        # requires_grad = False
        with torch.no_grad():
            # mask = (1 dot M) + 0 = M
            output_mask = self.mask_conv(mask)

        if self.input_conv.bias is not None:
            # spreads existing bias values out along 2nd dimension (channels) and then expands to output size
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        # mask_sum is the sum of the binary mask at every partial convolution location
        mask_is_zero = (output_mask == 0)
        # temporarily sets zero values to one to ease output calculation 
        mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum == 0
        output = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(mask_is_zero, 0.0)

        # mask is updated at each location
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

        if self.bn:
            output = self.batch_normalization(output)

        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_mask


class BatchNorm2d(nn.Module):
    def __init__(self, hidden_units):
        super(BatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(hidden_units)

    def forward(self, x):
        outs = self.bn(x[0])
        masks = self.bn(x[1])
        return outs, masks

if __name__ == '__main__':
    # img = torch.zeros(7, 4, 16, 64)
    # mask = torch.zeros(7, 3, 16, 64)
    # size = (3, 3, 16, 64)
    
    # inp = torch.ones(size)
    # input_mask = torch.ones(size)
    # input_mask[:, :, 100:, :][:, :, :, 100:] = 0
    # model = TSRN_PartialConv(mask=True)
    # print(model)
    # output = model([img, mask])
    img = torch.zeros(7, 4, 16, 64)
    torchvision.utils.save_image(img, "outputs.png")
