import torch
from torch import nn
import sys
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd

img_shape = (3, 32, 128)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 128)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    # fake_samples = fake_samples.to("cuda:0")
    # real_samples = real_samples.to("cuda:0")
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # interpolates = interpolates.to("cuda:0")
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Wasserstein Gradient Penalty Loss
class WGAN_GP_Loss(nn.Module):
    def __init__(self, lambda_gp=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(WGAN_GP_Loss, self).__init__()

        self.discriminator = Discriminator(img_shape=(3, 32, 128))
        self.discriminator = self.discriminator.to(device)
        self.compute_gradient_penalty = compute_gradient_penalty
        self.lambda_gp = lambda_gp
        
    def forward(self, fake_imgs, real_imgs):
        # Real images
        real_validity = self.discriminator(real_imgs)
        # Fake images
        fake_validity = self.discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs, fake_imgs)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

        return d_loss


if __name__ == '__main__':
    real_imgs = torch.zeros(7, 3, 32, 128)
    fake_imgs = torch.zeros(7, 3, 32, 128)
    real_imgs = real_imgs.to("cuda")
    fake_imgs = fake_imgs.to("cuda")
    wgan_loss = WGAN_GP_Loss(lambda_gp=10)
    loss = wgan_loss(fake_imgs, real_imgs)
    print("loss: ", loss)
    # d_loss.backward() 