import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.conv2 = ConvBlock(out_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x

class Darknet53(nn.Module):
    def __init__(self, num_blocks):
        super(Darknet53, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3, 1, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1)
        self.residual1 = self.make_layer(64, 32, num_blocks[0])
        self.conv3 = ConvBlock(64, 128, 3, 2, 1)
        self.residual2 = self.make_layer(128, 64, num_blocks[1])
        self.conv4 = ConvBlock(128, 256, 3, 2, 1)
        self.residual3 = self.make_layer(256, 128, num_blocks[2])
        self.conv5 = ConvBlock(256, 512, 3, 2, 1)
        self.residual4 = self.make_layer(512, 256, num_blocks[3])
        self.conv6 = ConvBlock(512, 1024, 3, 2, 1)
        self.residual5 = self.make_layer(1024, 512, num_blocks[4])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.conv3(x)
        x = self.residual2(x)
        x = self.conv4(x)
        x = self.residual3(x)
        x = self.conv5(x)
        x = self.residual4(x)
        x = self.conv6(x)
        x = self.residual5(x)
        return x

    def make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

class YOLOv3Watermark(nn.Module):
    def __init__(self):
        super(YOLOv3Watermark, self).__init__()
        self.darknet53 = Darknet53(num_blocks=[1, 2, 8, 8, 4])
        self.conv7 = ConvBlock(1024, 512, 1, 1, 0)
        self.conv8 = ConvBlock(512, 1024, 3, 1, 1)
        self.conv9 = ConvBlock(1024, 512, 1, 1, 0)
        self.conv10 = ConvBlock(512, 1024, 3, 1, 1)
        self.conv11 = ConvBlock(1024, 512, 1, 1, 0)
        self.conv12 = ConvBlock(512, 1024, 3, 1, 1)
        self.conv13 = nn.Conv2d(1024, 4, 1)  # Output layer for 4 bounding box coordinates

    def forward(self, x):
        x = self.darknet53(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        return x

def yolo_to_bbox_list(tensor, img_size=256, grid_size=8):
    """Convert the YOLO format tensor and return a list of tuples with (y1, x1, y2, x2) coordinates.
    If the batch_size in the DataLoader is 5, the returned list will have 5 bboxes, one for each item in the batch."""
    batch_size = tensor.size(0)
    bbox_list = []

    for b in range(batch_size):
        # Find the cell with the maximum sum of tx, ty, tw, and th
        max_sum, max_idx = torch.max(tensor[b].view(4, -1).sum(dim=0), dim=0)
        cy, cx = max_idx // grid_size, max_idx % grid_size

        # Get the values of the corresponding cell
        tx, ty, tw, th = tensor[b, :, cy, cx]

        # Convert tx, ty, tw, th to the original scale
        cx, cy = (cx + tx) * (img_size / grid_size), (cy + ty) * (img_size / grid_size)
        w, h = tw * img_size, th * img_size
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = x1 + w, y1 + h

        # ignore the ordering here, it works as "y1, x1, y2, x2" when you unpack it lol
        bbox_list.append((int(y1), int(y2), int(x1), int(x2)))

    return bbox_list

# custom loss func that should work with [batch_size, 4, 8, 8] tensors
def yolo_bbox_loss(pred, target, coord_scale=5):
    # Get the mask for the cells responsible for detecting an object
    obj_mask = target[..., 4] > 0
    
    # Compute the coordinate loss (x, y, w, h)
    coord_loss = (pred[..., :4] - target[..., :4]) ** 2
    coord_loss = torch.sum(coord_loss * obj_mask[..., None])

    # Apply the coord_scale
    coord_loss *= coord_scale

    return coord_loss