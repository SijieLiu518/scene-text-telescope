import torch
from torch import nn
import sys
sys.path.append('/home/videt/lsj/hat_textzoom/src')
from model.crnn import CRNN


class ContentPercptualLoss(nn.Module):
    def __init__(self, loss_weight=5e-4):
        super(ContentPercptualLoss, self).__init__()

        # ContentPercptualLoss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crnn = CRNN(32, 1, 37, 256)
        crnn = crnn.to(self.device)
        crnn_path = './dataset/TextZoom/crnn.pth'
        crnn.load_state_dict(torch.load(crnn_path))
        feature_map1 = nn.Sequential(crnn.cnn[:3]).eval()
        feature_map2 = nn.Sequential(crnn.cnn[3:6]).eval()
        feature_map3 = nn.Sequential(crnn.cnn[6:12]).eval()
        feature_map4 = nn.Sequential(crnn.cnn[12:18]).eval()
        feature_map5 = nn.Sequential(crnn.cnn[18:]).eval()
        
        for feature_map in [feature_map1, feature_map2, feature_map3, feature_map4, feature_map5]:
            for param in feature_map.parameters():
                param.requires_grad = False
        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.feature_map3 = feature_map3
        self.feature_map4 = feature_map4
        self.feature_map5 = feature_map5
        self.mse_loss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, out_images, target_images):
        out_images = out_images.to(self.device)
        target_images = target_images.to(self.device)
        # ContentPercptualLoss
        out = self.feature_map1(parse_crnn_data(out_images[:, :3, :, :]))
        target = self.feature_map1(parse_crnn_data(target_images[:, :3, :, :]))
        CP_loss = self.mse_loss(out, target)
        
        out = self.feature_map2(out)
        target = self.feature_map2(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map3(out)
        target = self.feature_map3(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map4(out)
        target = self.feature_map4(target)
        CP_loss += self.mse_loss(out, target)

        out = self.feature_map5(out)
        target = self.feature_map5(target)
        CP_loss += self.mse_loss(out, target)
        
        cp_loss = self.loss_weight*CP_loss
        
        return cp_loss


def parse_crnn_data(imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

if __name__ == '__main__':    
    loss = ContentPercptualLoss()
    out_images = torch.zeros(7, 3, 32, 128)
    target_images = torch.zeros(7, 3, 32, 128)
    loss(out_images, target_images)