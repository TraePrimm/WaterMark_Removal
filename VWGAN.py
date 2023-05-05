import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # encoder layers
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.self_attn = nn.MultiheadAttention(512, num_heads=8)
        
        # decoder layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
    
    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        b, c, h, w = x.shape
        x = torch.flatten(x, start_dim=2)  # flatten height and width dimensions
        x = x.permute(2, 0, 1)  # permute to (sequence_length, batch_size, embedding_dim)
        x, _ = self.self_attn(x, x, x)  # self-attention layer
        x = x.permute(1, 2, 0).view(b, c, h, w)  # reshape to (batch_size, num_channels, height, width)
        
        
        # decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # use sigmoid activation for pixel values
        
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=0)
    
    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = nn.functional.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = nn.functional.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = self.conv5(x)
        x = nn.functional.sigmoid(x)
        return x.squeeze()


class PixelNetwork(nn.Module):
    def __init__(self):
        super(PixelNetwork, self).__init__()

        # load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        # freeze VGG16 layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg16(x)
        return x




def ssim_loss(fake_images, real_images, win_size=3):
    fake_images_np = fake_images.detach().cpu().numpy()
    real_images_np = real_images.detach().cpu().numpy()

    # Calculate SSIM for each image in the batch
    batch_ssim = [ssim(fake_images_np[i], real_images_np[i], multichannel=True, win_size=win_size, data_range=1.0) for i in range(fake_images_np.shape[0])]

    # Convert the list of SSIM values to a torch Tensor
    ssim_loss = 1 - torch.Tensor(batch_ssim).mean()

    return ssim_loss

def vwgan_loss(pixel_network, real_outputs, fake_outputs, fake_images, real_images, lambda_l1=1.0, lambda_ssim=1.0, lambda_pixel=1.0):
    # Calculate Wasserstein distance loss
    wasserstein_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)
    
    # Calculate L1 loss
    l1_loss = F.l1_loss(fake_images, real_images)

    # Calculate SSIM loss
    ssim_loss_tensor = 1 - torch.mean(ssim_loss(fake_images, real_images))
    
    # Calculate PIXEL loss
    pixel_loss = 1 - torch.mean(ssim_loss(pixel_network(fake_images), pixel_network(real_images)))
    
    # Calculate total VWGAN loss
    total_loss = wasserstein_loss + lambda_l1 * l1_loss + lambda_ssim * ssim_loss_tensor + lambda_pixel * pixel_loss
    
    return total_loss