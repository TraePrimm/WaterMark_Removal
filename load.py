import math
import pathlib
from PIL import Image
import numpy as np
import os
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize


DATA_DIR = "wm-nowm"
TRAIN_VALID = ['train', 'valid']

def _check_files(data_dir):
    assert os.path.exists(data_dir), f"image dir '{data_dir}' does not exist"

    for tv in TRAIN_VALID:
            assert os.path.exists(os.path.join(data_dir, tv, 'no-watermark'))

def _list_files(data_dir):
    _check_files(data_dir)
    images = {}
    for tv in TRAIN_VALID:
        path = os.path.join(data_dir, tv, 'no-watermark')
        sub_paths = os.listdir(path)
        images.update((im, os.path.join(path, im)) for im in sub_paths)
    image_paths = list(images.values())
    image_paths.sort()

    return image_paths

def num_images(data_dir=DATA_DIR):
    return len(_list_files(data_dir))

def load_batched(batch_size, data_dir=DATA_DIR):
    N = num_images(data_dir)
    for i in range(0, N, batch_size):
        im_batch = load(slice(i, i+batch_size))
        yield im_batch

def load(s=slice(0,50), data_dir=DATA_DIR):
    _check_files(data_dir)

    images = []

    for im_path in _list_files(data_dir)[s]:
        im = Image.open(im_path)
        arr = np.array(im)
        assert np.issctype(arr.dtype)
        kind = arr.dtype.kind
        assert kind in ('u', 'f')
        if kind == 'u':
            max = np.iinfo(arr.dtype).max
            arr = arr.astype(np.float32) / max
        else:
            arr = arr.astype(np.float32)
        images.append(arr)

    return images

class WatermarkDataset(Dataset):
    NUM_WMS = 22

    VWGAN_MODE = (True, True, False, False, False)
    CNN_MODE = (False, False, True, True, False)
    YOLO_MODE = (False, True, False, False, True)
    def __init__(self, data_dir, mode, *, format_bb=False, scale_im=True, batch_size=1):
        self.data_dir = pathlib.Path(data_dir)
        self.full_no = self.data_dir.joinpath('full_no')
        self.full_wm = self.data_dir.joinpath('full_wm')
        self.no = self.data_dir.joinpath('no')
        self.wm = self.data_dir.joinpath('wm')
        self.paths = [self.full_no, self.full_wm, self.no, self.wm]
        self.bb_table = np.loadtxt(self.data_dir.joinpath('bb.csv'), delimiter=',', dtype=np.int32)
        self.mode = mode
        self.transforms = Resize((256, 256))
        self.format_bb = format_bb # set this True when you want to return bounding boxes with shape (4, 8, 8)
        self.scale_im = scale_im
        self.batch_size = batch_size

        self.by_wm = [[] for _ in range(WatermarkDataset.NUM_WMS)]
        for row in self.bb_table:
            self.by_wm[row[1]].append(row)

        self.len = int(math.ceil(min(len(wm_list) for wm_list in self.by_wm) * WatermarkDataset.NUM_WMS / batch_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index > self.len:
            raise IndexError
        wm_id = index % WatermarkDataset.NUM_WMS
        offset = index // WatermarkDataset.NUM_WMS
        stride = self.batch_size

        items = [self.load_row(row) for row in self.by_wm[wm_id][offset*stride:(offset+1)*stride]]
        transpose = tuple(torch.stack(col) for col in zip(*items))
        return transpose

    def load_row(self, row):
        im_id = row[0]
        img_name = f'{im_id}.jpg'

        # original code
        # data = tuple(self.load_from_path(path.joinpath(img_name)) for path, enabled in zip(self.paths, self.mode) if enabled)

        # image's original size has the following info: (Channels, height, width)
        original_size = None
        data = []
        for path, enabled in zip(self.paths, self.mode):
            if not enabled:
                continue

            image, original_size = self.load_from_path(path.joinpath(img_name))
            data.append(image)

        # scale factor for bounding box resizing
        _, height, width = original_size
        scale_factor_x = 256 / width
        scale_factor_y = 256 / height

        data = tuple(data)
        if self.mode[-1]:
            # this bbox is in (y1, x1, y2, x2) format (?)
            # the images will have been transformed using Resize((256, 256)), but the bbox wasn't, try to reshape it...
            bbox = row[2:]
            if self.scale_im:
                x1_resized = int(bbox[1] * scale_factor_x)
                x2_resized = int(bbox[3] * scale_factor_x)
                y1_resized = int(bbox[0] * scale_factor_y)
                y2_resized = int(bbox[2] * scale_factor_y)
                bbox = torch.tensor([y1_resized, y2_resized, x1_resized, x2_resized])
            else:
                bbox = torch.tensor([bbox[0], bbox[2], bbox[1], bbox[3]])


            if not self.format_bb:
                # concatenate the bb as is, in default (y1, y2, x1, x2) format. The bb will have a shape of [1, 4] when used w/ dataloader
                data = data + (bbox,)
            else:
                # convert the bbox to [1, 4, 8, 8] format for loss function calculation
                target_tensor = self.process_bbox(bbox)
                data = data + (target_tensor,)


        return data

    def process_bbox(self, bbox):
        # parameter bbox is in (y1, y2, x1, x2) format, directly from the csv.
        # format the bounding boxes to YOLO format. This will have a shape of [1, 4, 8, 8]
        y1, y2, x1, x2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        # Normalize and scale target bounding boxes
        cx, cy = cx / (256 / 8), cy / (256 / 8)  # Assuming 256x256 images and 8x8 grid
        w, h = w / 256, h / 256  # Normalize width and height

        # Initialize target tensor with zeros and set the corresponding cell values
        target_tensor = torch.zeros((4, 8, 8), dtype=torch.float32)
        x_min_cell, x_max_cell = max(0, int(cx - w / 2 * 8)), min(8, int(cx + w / 2 * 8) + 1)
        y_min_cell, y_max_cell = max(0, int(cy - h / 2 * 8)), min(8, int(cy + h / 2 * 8) + 1)

        for ix in range(x_min_cell, x_max_cell):
            for iy in range(y_min_cell, y_max_cell):
                target_tensor[:, iy, ix] = torch.tensor([cx - ix, cy - iy, w, h])

        return target_tensor


    def load_from_path(self, path):
        image = read_image(str(path), ImageReadMode.RGB)
        orig_size = image.shape

        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        elif torch.is_floating_point(image.dtype) and image.dtype != torch.float32:
            image = image.to(torch.float32)
        assert image.dtype == torch.float32, f'Type is {image.dtype}'
        if self.scale_im:
            image = self.transforms(image)
        # return image
        return image, orig_size

class StockImageDataset(Dataset):
    def __init__(self, data_dir, num_images):
        self.data_dir = data_dir
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        nwm_image_path = f'{self.data_dir}/{idx}_nwm.jpg'
        wm_image_path = f'{self.data_dir}/{idx}_wm.jpg'
        return (
            self.load_from_path(wm_image_path),
            self.load_from_path(nwm_image_path)
        )

    def load_from_path(self, path):
        image = read_image(path, ImageReadMode.RGB)
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        elif torch.is_floating_point(image.dtype) and image.dtype != torch.float32:
            image = image.to(torch.float32)
        assert image.dtype == torch.float32, f'Type is {image.dtype}'
        return image