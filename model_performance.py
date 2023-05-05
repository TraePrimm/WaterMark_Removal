# model_performance.py evaluates the performance of the GAN and UNET
# using various image comparison metrics from skimage
import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from load import WatermarkDataset
from unet import UNet
from VWGAN import Generator, Discriminator
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, CenterCrop
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

# TODO
UNET_MODEL_PATH = 'Project/Github_05022023/unet_gh/unet_training_1_80_sam.pt'
VWGAN_GENERATOR_MODEL_PATH = 'Project/Github_05022023/generator_fixed_loss_1.pt' #generator_fixed_loss_1.pt
DATA_DIR = "Project/Github_05022023/logos_rendered"
SET = "Train"

def load_from_path_GAN(path):
        image = read_image(str(path), ImageReadMode.RGB)

        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        elif torch.is_floating_point(image.dtype) and image.dtype != torch.float32:
            image = image.to(torch.float32)
        assert image.dtype == torch.float32, f'Type is {image.dtype}'
        # return image
        return image

def load_from_path(path):
    image = read_image(str(path), ImageReadMode.RGB)
    orig_size = image.shape

    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    elif torch.is_floating_point(image.dtype) and image.dtype != torch.float32:
        image = image.to(torch.float32)
    assert image.dtype == torch.float32, f'Type is {image.dtype}'
    # return image
    scale_height = 400 / orig_size[1]
    scale_width = 400 / orig_size[2]
    scale = min(scale_height, scale_width)
    resize_dim = tuple(int(dim * scale) for dim in orig_size[1:3])
    transform = Resize(resize_dim, antialias=True)
    crop_dim = tuple(dim - (dim % 16) for dim in resize_dim)
    cropping = CenterCrop(crop_dim)
    return cropping(transform(image))

def evaluate_performance(model: nn.Module, imageFolder, device: str, metric: str, mode: str):
    model.eval() # set model to evaluate mode
    total_metric_value = 0.0
    with torch.no_grad():
        
        if(SET == "Test"):
            imageFolder += "/test"
            full_nwm_image_path = f'{imageFolder}/no/'
            full_wm_image_path = f'{imageFolder}/wm/'  # Same as the small because I'm not cropping that
            nwm_image_path = f'{imageFolder}/no/'
            wm_image_path = f'{imageFolder}/wm/'

        if(SET == "Train"):
            full_nwm_image_path = f'{imageFolder}/full_no/'
            full_wm_image_path = f'{imageFolder}/full_wm/'

            nwm_image_path = f'{imageFolder}/no/'
            wm_image_path = f'{imageFolder}/wm/'

        for idx in range(10):
            # try:
            if(mode == "CNN"):
                # CNN image
                wm_image = load_from_path(wm_image_path + f"{idx}.jpg")
                nwm_image = load_from_path(nwm_image_path + f"{idx}.jpg")
                wm_image.cpu().numpy().transpose(1, 2, 0)
                output = model.forward(wm_image.to(device).unsqueeze(0))
                output = torch.squeeze(output, dim=0)
                output.cpu().numpy().transpose(1, 2, 0)

            #GAN image
            if(mode == "GAN"):
                # wm_image = cv2.imread(wm_image_path + f"{idx}.jpg")
                # nwm_image = cv2.imread(nwm_image_path + f"{idx}.jpg")
                wm_image = load_from_path_GAN(full_wm_image_path + f"{idx}.jpg")
                nwm_image = load_from_path_GAN(full_nwm_image_path + f"{idx}.jpg")

                # wm_image = torch.from_numpy(wm_image)
                # nwm_image = torch.from_numpy(nwm_image)

                nwm_image = nwm_image.to(device)
                wm_image = wm_image.to(device)

                transform = Resize((256, 256), antialias=True)
                wm_image = transform(wm_image)
                nwm_image = transform(nwm_image)

                output = model.forward(wm_image.unsqueeze(0))
                output = torch.squeeze(output, dim=0)
            # except:
            #     pass

            # wm_image = (wm_image - np.min(wm_image)) / (np.max(wm_image) - np.min(wm_image))
            # nwm_image = (nwm_image - np.min(nwm_image)) / (np.max(nwm_image) - np.min(nwm_image))
            # output = (output - np.min(output)) / (np.max(output) - np.min(output))

            wm3 = np.moveaxis(wm_image.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            nwm3 = np.moveaxis(nwm_image.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            tensor_image3 = np.moveaxis(output.cpu().numpy(), (0, 1, 2), (2, 0, 1))

            if(idx in (0, 2, 4, 6)):
                f, ax = plt.subplots(1,3)
                plt.suptitle(f"{mode}")
                ax[0].imshow(wm3)
                ax[0].set_title('Watermarked Image')
                ax[0].axis('off')
                ax[1].imshow(nwm3)
                ax[1].set_title('Ground Truth')
                ax[1].axis('off')
                ax[2].imshow(tensor_image3)
                ax[2].set_title('Reconstructed Image')
                ax[2].axis('off')

                plt.show()
                plt.close()

            if metric == "mse":
                # MSE measures avg. squared distance between the pixel values of the images
                # note: torch has mse loss in nn.functional which uses tensors
                # however, the other metrics are not in default torch (ignite has them though)
                # print(f"MODE = {mode} -> nwm shape: {nwm_image.shape} output shape: {output.shape}")
                metric_value = mean_squared_error(nwm_image.cpu().numpy(), output.cpu().numpy())
            elif metric == "psnr":
                # compares the maximum power of a signal to the corrupting noise
                # higher PSNR can indicate better image quality
                if(mode == "GAN"):
                    wm_image /= 255
                    nwm_image /= 255
                    output /= 255
                metric_value = peak_signal_noise_ratio(nwm_image.cpu().numpy(), output.cpu().numpy())
                # metric_value = peak_signal_noise_ratio(nwm_image.cpu().numpy(), output.cpu().numpy())
            elif metric == "ssim":
                # considers structural information from the images along with luminance, contrast etc.
                # basically, ssim takes image texture into account
                # TODO: make sure the max of the data is being passed
                metric_value = structural_similarity(nwm_image.cpu().numpy(), output.cpu().numpy(), data_range=255, multichannel=True, channel_axis=0)
                # metric_value = structural_similarity(nwm_image.cpu().numpy(), output.cpu().numpy(), multichannel=True)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            total_metric_value += metric_value

    avg_metric_value = total_metric_value / 10
    return avg_metric_value

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = WatermarkDataset(DATA_DIR, WatermarkDataset.CNN_MODE, scale_im=False)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # Load the trained models

    unet_model = UNet(4).to(device)
    sd = torch.load(UNET_MODEL_PATH)
    unet_model.load_state_dict(sd['model'])

    # unet_model = UNet().to(device)
    # unet_model.load_state_dict(torch.load(UNET_MODEL_PATH))

    generator_model = Generator().to(device)
    generator_model.load_state_dict(torch.load(VWGAN_GENERATOR_MODEL_PATH))

    # Evaluate performance
    for metric in ["mse", "psnr", "ssim"]:
        print(f'unet {metric} score: {evaluate_performance(unet_model, DATA_DIR, device, metric, "CNN")}')
        print(f'GAN {metric} score: {evaluate_performance(generator_model, DATA_DIR, device, metric, "GAN")}\n\n')

if __name__ == "__main__":
    main()

