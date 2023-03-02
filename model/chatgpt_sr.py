import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PartialConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True): 
        super(PartialConv, self).__init__() 
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) 
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False) 
        self.input_conv.weight.data.normal_(0.0, 0.02) 
        self.mask_conv.weight.data.fill_(1.0)

        # Mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class SuperResolutionNet(nn.Module): 
    def __init__(self, upscale_factor): 
        super(SuperResolutionNet, self).__init__() 
        self.upscale_factor = upscale_factor

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.pc1 = PartialConv(3, 64, (5, 5), (1, 1), (2, 2))
        self.pc2 = PartialConv(64, 64, (3, 3), (1, 1), (1, 1))
        self.pc3 = PartialConv(64, 32, (3, 3), (1, 1), (1, 1))
        self.pc4 = PartialConv(32, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))

    def forward(self, x, mask):
        x = x[:, :3, :, :]
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.pixel_shuffle(self.conv4(out))
           
        out_pc, mask = self.pc1(x, mask)
        out_pc = self.relu(out_pc)
        out_pc, mask = self.pc2(out_pc, mask)
        out_pc = self.relu(out_pc)
        out_pc, mask = self.pc3(out_pc, mask)
        out_pc = self.relu(out_pc)
        out_pc, mask = self.pc4(out_pc, mask)
        out_pc = self.pixel_shuffle(out_pc)

        return out + out_pc, mask
    
if __name__ == '__main__':
    img = torch.zeros(7, 3, 16, 64)
    mask = torch.zeros(7, 3, 16, 64)
    net = SuperResolutionNet(upscale_factor=2)
    output, mask = net(img, mask)
    print(output.size())
    
    
    