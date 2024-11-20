import numpy as np
import torch
from torch.utils import data
import skimage.io
import glob

def indexGenerate(x_start, y_start, p, size):
    xs = torch.linspace(x_start, x_start + p - 1, steps=p)
    ys = torch.linspace(y_start, y_start + p - 1, steps=p)
    y, x = torch.meshgrid(ys, xs, indexing='ij')  
    x_norm = x / size
    y_norm = y / size
    final = torch.stack((x_norm, y_norm), dim=2)  
    return final

class CM2Dataset(data.Dataset):
    def __init__(self, dir_data, transform=None, lens_centers=None):
        self.dir_data = dir_data
        self.transform = transform
        self.lens_centers = lens_centers
        self.num_stacks = len(glob.glob(self.dir_data + '/demix_*.tif'))// 3  # Number of image stacks
        self.image_size = 2400  

        # Load the mask
        mask_path = './masked_mask_rp_0.tif'
        self.mask = skimage.io.imread(mask_path).astype('float32') / 65535.0  # Normalize uint16 mask to [0, 1]

    def __getitem__(self, index):
        meas_path = f'{self.dir_data}/demix_{index * 3 + 1}.tif'
        meas_stack = skimage.io.imread(meas_path)
        meas_stack = meas_stack.astype('float32') / meas_stack.max()

        # Apply the mask
        meas_stack *= self.mask

        # Normalize again after applying the mask
        meas_stack = meas_stack / meas_stack.max()

        if meas_stack.ndim == 3 and meas_stack.shape[2] == 9:
            meas = meas_stack  # [H, W, 9]
        elif meas_stack.ndim == 3 and meas_stack.shape[0] == 9:
            meas = meas_stack.transpose(1, 2, 0)  # [H, W, 9]
        else:
            raise ValueError(The shape of meas_stack in file {index * 3 + 1} does not match the expected shape: {meas_stack.shape})
        gt_path = f'{self.dir_data}/gt_{index * 3 + 1}.tif'
        gt = skimage.io.imread(gt_path)
        gt = gt.astype('float32') / gt.max()

        index_list = []
        for loc in self.lens_centers:
            x_center, y_center = loc
            x_start = x_center - self.image_size // 2
            y_start = y_center - self.image_size // 2
            pixel_coords = indexGenerate(x_start, y_start, self.image_size, self.image_size)  # [H, W, 2]
            index_list.append(pixel_coords)

        index_list = torch.stack(index_list, dim=0)  # [9, H, W, 2]

        meas = torch.from_numpy(meas).permute(2, 0, 1)  # [9, H, W]

        gt = torch.from_numpy(gt).unsqueeze(0)  # [1, H, W]

        data = {'gt': gt, 'meas': meas, 'index': index_list}

        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.num_stacks

class Subset(data.Dataset):
    def __init__(self, dataset, isVal, patch_size=480, stride=240):

        self.dataset = dataset
        self.isVal = isVal
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = 2400

        if not isVal:
            self.patches_per_row = (self.image_size - self.patch_size) // self.stride + 1
            self.patches_per_image = self.patches_per_row * self.patches_per_row
            self.total_patches = len(self.dataset) * self.patches_per_image
        else:
            self.total_patches = len(self.dataset)

    def __getitem__(self, index):
        if self.isVal:
            data = self.dataset[index]
            return data
        else:
            stack_index = index // self.patches_per_image
            patch_index = index % self.patches_per_image

            data = self.dataset[stack_index]
            gt, meas, index_list = data['gt'], data['meas'], data['index']

            row = patch_index // self.patches_per_row
            col = patch_index % self.patches_per_row
            start_y = row * self.stride
            start_x = col * self.stride

            gt_patch = gt[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

            meas_patch = meas[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

            index_patch = index_list[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size, :]

            data = {'gt': gt_patch, 'meas': meas_patch, 'index': index_patch}
            if self.dataset.transform:
                data = self.dataset.transform(data)
            return data

    def __len__(self):
        if self.isVal:
            return self.dataset.__len__()
        else:
            return self.total_patches

class ToTensorcm2(object):

    def __call__(self, data):
        gt, meas, index = data['gt'], data['meas'], data['index']
        gt = gt.float()  # [1, H, W]
        meas = meas.float()  # [9, H, W]
        index = index.float()  # [9, H, W, 2]
        return {'gt': gt,
                'meas': meas,
                'index': index
                }
