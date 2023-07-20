import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional
from typing import Optional


class RandomImagePixelationDataset(Dataset):
    def __init__(self, image_dir, width_range: tuple[int, int], height_range: tuple[int, int], size_range: tuple[int, int],dtype: Optional[type] = None,):
        self.image_files = []
        directories = os.listdir(image_dir)
        for subdir in directories:
            if subdir == '.DS_Store':
                continue
            for name in os.listdir(os.path.join(image_dir, subdir)):
                self.image_files.append(os.path.join(image_dir, subdir, name))
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        self.data_transforms = transforms.Compose([
            transforms.Resize(size=64, interpolation=functional.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=64),
            transforms.Grayscale(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]

        with Image.open(image_file) as img:
            img = self.data_transforms(img)
            image = np.array(img, dtype=self.dtype)

        width = np.random.randint(self.width_range[0], self.width_range[1] + 1)
        height = np.random.randint(self.height_range[0], self.height_range[1] + 1)
        size = np.random.randint(self.size_range[0], self.size_range[1] + 1)
        x = np.random.randint(0, image.shape[1] - width + 1)
        y = np.random.randint(0, image.shape[0] - height + 1)

        pixelated_image, known_array, target_array = self.prepare_image(image, x, y, width, height, size)

        pixelated_image = functional.to_tensor(pixelated_image)
        known_array = functional.to_tensor(known_array)
        target_array = functional.to_tensor(target_array)

        return pixelated_image, known_array, target_array

    @staticmethod
    def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray]:
        if width < 2 or height < 2 or size < 2:
            raise ValueError("width/height/size must be >= 2")
        if x < 0 or (x + width) > image.shape[-1]:
            raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
        if y < 0 or (y + height) > image.shape[-2]:
            raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
        target_array = image.copy()

        area = (..., slice(y, y + height), slice(x, x + width))

        pixelated_image = RandomImagePixelationDataset.pixelate(image, x, y, width, height, size)

        known_array = np.ones_like(image, dtype=bool)
        known_array[area] = False
        return pixelated_image, known_array, target_array

    @staticmethod
    def pixelate(image, x, y, width, height, size):
        curr_x = x
        while curr_x < x + width:
            curr_y = y
            while curr_y < y + height:
                block = (slice(curr_y, min(curr_y + size, y + height)),
                         slice(curr_x, min(curr_x + size, x + width)))
                image[block] = np.mean(image[block])
                curr_y += size
            curr_x += size

        return image


def stack_with_padding(batch_as_list: list):
    n = len(batch_as_list)
    shapes = [item[0].shape for item in batch_as_list]
    max_shape = tuple(max(dim) for dim in zip(*shapes))
    stacked_pixelated = torch.zeros((n, *max_shape), dtype=torch.float32)
    stacked_known = torch.ones((n, *max_shape), dtype=torch.float32)
    stacked_target = torch.zeros((n, *max_shape), dtype=torch.float32)

    for i, (pixelated_image, known_array, target_array) in enumerate(batch_as_list):
        channels, height, width = pixelated_image.shape
        stacked_pixelated[i, :channels, :height, :width] = pixelated_image
        stacked_known[i, :channels, :height, :width] = known_array
        stacked_target[i, :channels, :height, :width] = target_array

    return stacked_pixelated, stacked_known, stacked_target

class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.pixelated_images = torch.from_numpy(np.array(data['pixelated_images']))
        self.known_arrays = torch.from_numpy(np.array(data['known_arrays']))

    def __len__(self):
        return len(self.pixelated_images)

    def __getitem__(self, idx):
        pixelated_image = self.pixelated_images[idx]
        known_array = self.known_arrays[idx]
        return pixelated_image, known_array












