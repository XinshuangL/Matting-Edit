import os
import math
import numbers
import random
import logging
import numpy as np
import imgaug.augmenters as iaa

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms

from   utils import CONFIG

from random import randint

import warnings
warnings.filterwarnings("ignore")
import cv2
interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase
        
    def get_box_from_alpha(self, alpha_final):
        bi_mask = np.zeros_like(alpha_final)
        bi_mask[alpha_final>0.5] = 1
        fg_set = np.where(bi_mask != 0)

        if len(fg_set[1]) == 0 or len(fg_set[0]) == 0:
            x_min = random.randint(1, 511)
            x_max = random.randint(1, 511) + x_min
            y_min = random.randint(1, 511)
            y_max = random.randint(1, 511) + y_min
        else:
            ok = False
            while_times = 0
            while not ok:
                if while_times >= 10:
                    x_min = random.randint(1, 511)
                    x_max = random.randint(1, 511) + x_min
                    y_min = random.randint(1, 511)
                    y_max = random.randint(1, 511) + y_min
                    break
                
                x_min = np.min(fg_set[1]) + random.randint(-100, 100)
                x_max = np.max(fg_set[1]) + random.randint(-100, 100)
                y_min = np.min(fg_set[0]) + random.randint(-100, 100)
                y_max = np.max(fg_set[0]) + random.randint(-100, 100)
                
                x_min = max(x_min, 0)
                x_max = min(x_max, 1022)
                y_min = max(y_min, 0)
                y_max = min(y_max, 1022)

                ok = (x_min < x_max) and (y_min < y_max)
                while_times += 1                

        bbox = np.array([x_min, y_min, x_max, y_max])
        return bbox

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha1, trimap1, alpha2, trimap2 = sample['image'][:,:,::-1], sample['alpha1'],  sample['trimap1'], sample['alpha2'],  sample['trimap2']
        
        alpha1[alpha1 < 0 ] = 0
        alpha1[alpha1 > 1] = 1
        alpha2[alpha2 < 0 ] = 0
        alpha2[alpha2 > 1] = 1
        
        bbox1 = self.get_box_from_alpha(alpha1)
        bbox2 = self.get_box_from_alpha(alpha2)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha1 = np.expand_dims(alpha1.astype(np.float32), axis=0)
        alpha2 = np.expand_dims(alpha2.astype(np.float32), axis=0)
        trimap1[trimap1 < 85] = 0
        trimap1[trimap1 >= 170] = 2
        trimap1[trimap1 >= 85] = 1
        trimap2[trimap2 < 85] = 0
        trimap2[trimap2 >= 170] = 2
        trimap2[trimap2 >= 85] = 1

        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            fg1 = sample['fg1'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg1'] = torch.from_numpy(fg1).sub_(self.mean).div_(self.std)
            fg2 = sample['fg2'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg2'] = torch.from_numpy(fg2).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
        
        sample['boxes1'] = torch.from_numpy(bbox1).to(torch.float)[None,...]
        sample['boxes2'] = torch.from_numpy(bbox2).to(torch.float)[None,...]

        sample['image'], sample['alpha1'], sample['trimap1'], sample['alpha2'], sample['trimap2'] = torch.from_numpy(image), torch.from_numpy(alpha1), torch.from_numpy(trimap1).to(torch.long), torch.from_numpy(alpha2), torch.from_numpy(trimap2).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        if CONFIG.model.trimap_channel == 3:
            sample['trimap1'] = F.one_hot(sample['trimap1'], num_classes=3).permute(2,0,1).float()
            sample['trimap2'] = F.one_hot(sample['trimap2'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap1'] = sample['trimap1'][None,...].float()
            sample['trimap2'] = sample['trimap2'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        return sample

class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    def process_one(self, fg, alpha):
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        return fg, alpha

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg1, alpha1, fg2, alpha2 = sample['fg1'], sample['alpha1'], sample['fg2'], sample['alpha2']

        sample['fg1'], sample['alpha1'] = self.process_one(fg1, alpha1)
        sample['fg2'], sample['alpha2'] = self.process_one(fg2, alpha2)

        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def apply2fg(self, fg, alpha):
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        return fg*255

    def __call__(self, sample):
        fg1, alpha1 = sample['fg1'], sample['alpha1']
        if not np.all(alpha1==0):
            sample['fg1'] = self.apply2fg(fg1, alpha1)
        fg2, alpha2 = sample['fg2'], sample['alpha2']
        if not np.all(alpha2==0):
            sample['fg2'] = self.apply2fg(fg2, alpha2)
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def crop(self, fg, alpha, trimap):
        h, w = trimap.shape
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        return fg_crop, alpha_crop, trimap_crop

    def __call__(self, sample):
        fg1_crop, alpha1_crop, trimap1_crop = self.crop(sample['fg1'],  sample['alpha1'], sample['trimap1'])
        fg2_crop, alpha2_crop, trimap2_crop = self.crop(sample['fg2'],  sample['alpha2'], sample['trimap2'])

        bg = sample['bg']
        ratio = 1.1*max(self.output_size[0]/bg.shape[0], self.output_size[1]/bg.shape[1])
        bg = cv2.resize(bg, (int(bg.shape[1]*ratio), int(bg.shape[0]*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))

        left_top = (random.randint(0, bg.shape[0]-self.output_size[0]), random.randint(0, bg.shape[1]-self.output_size[1]))
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]

        if len(np.where(sample['trimap1']==128)[0]) == 0 or len(np.where(sample['trimap2']==128)[0]) == 0:
            fg1_crop = cv2.resize(sample['fg1'], self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha1_crop = cv2.resize(sample['alpha1'], self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap1_crop = cv2.resize(sample['trimap1'], self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            fg2_crop = cv2.resize(sample['fg2'], self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2_crop = cv2.resize(sample['alpha2'], self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap2_crop = cv2.resize(sample['trimap2'], self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        
        sample.update({
            'fg1': fg1_crop, 'alpha1': alpha1_crop, 'trimap1': trimap1_crop, 
            'fg2': fg2_crop, 'alpha2': alpha2_crop, 'trimap2': trimap2_crop, 
            'bg': bg_crop})
        return sample

class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def alpha2trimap(self, alpha):
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        return trimap

    def __call__(self, sample):
        sample['trimap1'] = self.alpha2trimap(sample['alpha1'])
        sample['trimap2'] = self.alpha2trimap(sample['alpha2'])
        return sample

def shrink(fg, alpha, mode):
    H, W, _ = fg.shape
    if random.randint(0, 1) == 0:
        H_len = round(H * 0.75)
        W_len = round(W * 0.75)
    else:
        H_len = round(H * 2/3)
        W_len = round(W * 2/3)
    
    if mode in [0, 1]:
        H_low = 0
        H_high = H_len
    else:
        H_low = H - H_len
        H_high = H

    if mode in [0, 2]:
        W_low = 0
        W_high = W_len
    else:
        W_low = W - W_len
        W_high = W
        
    new_fg = np.zeros_like(fg, dtype=np.float32)
    new_alpha = np.zeros_like(alpha, dtype=np.float32)

    fg = cv2.resize(fg, (W_len, H_len))
    alpha = cv2.resize(alpha, (W_len, H_len))

    new_fg[H_low:H_high,W_low:W_high] = fg
    new_alpha[H_low:H_high,W_low:W_high] = alpha

    return new_fg, new_alpha

class Composite(object):
    def __call__(self, sample):
        fg1, alpha1, fg2, alpha2, bg = sample['fg1'], sample['alpha1'], sample['fg2'], sample['alpha2'], sample['bg']
        alpha1[alpha1 < 0 ] = 0
        alpha1[alpha1 > 1] = 1
        alpha2[alpha2 < 0 ] = 0
        alpha2[alpha2 > 1] = 1
        fg1[fg1 < 0 ] = 0
        fg1[fg1 > 255] = 255
        fg2[fg2 < 0 ] = 0
        fg2[fg2 > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        mode1, mode2 = random.sample([0, 1, 2, 3], 2)
        fg1, alpha1 = shrink(fg1, alpha1, mode1)
        fg2, alpha2 = shrink(fg2, alpha2, mode2)
        sample['fg1'] = fg1
        sample['alpha1'] = alpha1
        sample['fg2'] = fg2
        sample['alpha2'] = alpha2

        alpha2_occlude = alpha2 * (1 - alpha1)
        alpha_bg = (1 - alpha1) * (1 - alpha2)
        image = fg1 * alpha1[:, :, None] + fg2 * alpha2_occlude[:, :, None] + bg * alpha_bg[:, :, None]

        sample['image'] = image
        return sample
