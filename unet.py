import torch
from torch import nn
from torchvision import models

class UNetConv(nn.Module):
    def __init__(self, channel_tup):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_tup[0], channel_tup[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_tup[1])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_tup[1], channel_tup[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_tup[2])
        self.relu2 = nn.ReLU()

        self.conv = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2
            )

    def forward(self, x):
        return self.conv(x)

def UpConv(in_chan, out_chan):
        return nn.ConvTranspose2d(in_chan, out_chan, (2, 2), 2)

class UNet(nn.Module):
    def __init__(self, num_levels=5, init_conv=64, im_channels=3):
        super().__init__()
        assert num_levels >= 2
        self.down_stages = []
        self.max_pools = []
        self.up_stages = []
        self.up_convs = []

        self.down_stages.append(UNetConv((im_channels, init_conv, init_conv)))
        self.max_pools.append(nn.MaxPool2d((2, 2)))
        n_chan = init_conv

        for _ in range(num_levels - 2):
            self.down_stages.append(UNetConv((n_chan, n_chan*2, n_chan*2)))
            self.max_pools.append(nn.MaxPool2d((2, 2)))
            n_chan *= 2

        self.bottom = UNetConv((n_chan, n_chan*2, n_chan*2))

        for _ in range(num_levels - 2):
            self.up_convs.append(UpConv(n_chan*2, n_chan))
            self.up_stages.append(UNetConv((n_chan*2, n_chan, n_chan)))
            n_chan //= 2

        self.up_convs.append(UpConv(n_chan*2, n_chan))
        self.up_stages.append(UNetConv((n_chan*2, n_chan, n_chan)))

        self.out_conv = nn.Conv2d(n_chan, im_channels, (1, 1))

        self.modules = nn.ModuleList(self.down_stages + self.max_pools + self.up_stages + self.up_convs)


    def forward(self, x):
        intermediate = []
        # print('Init:', x.shape)
        for down, mp in zip(self.down_stages, self.max_pools):
            x = down.forward(x)
            # print('Left Conv:', x.shape)
            intermediate.append(x)
            x = mp(x)
            # print('Mp:', x.shape)
        x = self.bottom.forward(x)
        # print('Bottom Conv:', x.shape)
        for up, uc, across in zip(self.up_stages, self.up_convs, reversed(intermediate)):
            x = uc.forward(x)
            # print('Uc:', x.shape, across.shape)
            x = up.forward(torch.concat((x, across), 1))
            # print('Up:', x.shape)
        x = self.out_conv(x)
        # print('Out:', x.shape)
        return x

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        # freeze VGG16 layers
        for param in self.vgg16.parameters():
            param.requires_grad = False


    def forward(self, fake_images, true_images):
        # parse fake images through VGG16 and store output tensors
        fake_output = self.vgg16(fake_images)

        # parse true images through VGG16 and store output tensors
        true_output = self.vgg16(true_images)

        # calculate average difference in each convolutional layer
        return (fake_output - true_output).pow(2).mean()