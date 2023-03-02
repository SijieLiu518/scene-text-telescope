import torch
import torchvision.utils as vutils

# 创建一个64x64x28x28的feature map
features = torch.randn(64,64,28,28)

features = features.permute(1, 2, 0, 3) 
# 使用 make_grid 将特征图可视化
grid = vutils.make_grid(features)

# 保存可视化结果到文件
vutils.save_image(grid, 'featuremap.png')