import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/videt/lsj_SR/scene-text-telescope')
from model.ImageDiscriminator import ImageDiscriminator


class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the targe is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        # print("input", input.mean())
        
        return loss if is_disc else loss * self.loss_weight


def gradient_penalty_loss(discriminator, real_data, fake_data, mask=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpaitting. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1)).cuda()
    
    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gradients_penalty


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for wgan-gp.
    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.):
        super(GradientPenaltyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, discriminator, real_data, fake_data, mask=None):
        """
        Args:
            discriminator (nn.Module): Network for the discriminator.
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            mask (Tensor): Masks for inpaitting. Default: None.
        Returns:
            Tensor: Loss.
        """
        loss = gradient_penalty_loss(
            discriminator, real_data, fake_data, mask=mask)

        # return loss * self.loss_weight
        return loss * 10.0

if __name__ == '__main__':
    cri_gan = GANLoss(
        'wgan', 
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1e-6)
    cri_gan = cri_gan.to("cuda")
    
    cri_grad_penalty = GradientPenaltyLoss(
        loss_weight=10.0)
    cri_grad_penalty = cri_grad_penalty.to("cuda")
    
    # cri_grad_penalty = GradientPenaltyLoss(
    #     loss_weight='!!float 10.0'
    # )
    
    sr = torch.zeros(7, 3, 16, 64).to("cuda")
    gt = torch.zeros(7, 3, 16, 64).to("cuda")
    # ———————— net_d ————————
    net_d = ImageDiscriminator()
    net_d = net_d.to("cuda")
    # optimizers
    if net_d:
        weight_decay_d = 0
        optimizer_d = torch.optim.Adam(
            net_d.parameters(),
            lr=1e-4,
            weight_decay=weight_decay_d,
            betas=[0.9, 0.999])
        # self.optimizers.append(self.optimizer_d)
    
    # train net_d
    optimizer_d.zero_grad()
    for p in net_d.parameters():
        p.requires_grad = True
    
    real_d_pred = net_d(gt)
    fake_d_pred = net_d(sr.detach())
    
    l_d_real = cri_gan(real_d_pred, True, is_disc=True)
    l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)
    
    l_d_total = l_d_real + l_d_fake
    
    l_grad_penalty = cri_grad_penalty(
        net_d, gt, sr)
    # self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
    l_d_total += l_grad_penalty
    
    optimizer_d.step()
    
    l_g_total = 0
    if net_d:
        # gan loss
        fake_g_pred = net_d(sr)
        l_g_gan = cri_gan(fake_g_pred, True, is_disc=False)
        l_g_total += l_g_gan
    
    print(l_d_total)
    print(l_g_gan)
    
    
