import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb,m,mask):
        for op in self.ops:
            rgb,m, mask = op(rgb,m, mask)
        return rgb,m, mask

class Normalize(object):
    def __init__(self, mean1,mean2, std1,std2):
        self.mean1 = mean1
        self.mean2 = mean2
        self.std1 = std1
        self.std2 = std2

    def __call__(self, rgb,m, mask):
        rgb = (rgb - self.mean1)/self.std1
        m = (m - self.mean2) / self.std2
        mask /= 255
        return rgb,m, mask

class Minusmean(object):
    def __init__(self, mean1,mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, rgb,m, mask):
        rgb = rgb - self.mean1
        m = m - self.mean2
        mask /= 255
        return rgb,m, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,m, mask):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb, m, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,m, mask):
        H,W,_ = rgb.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        rgb = rgb[ymin:ymin+self.H, xmin:xmin+self.W, :]
        m = m[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return rgb, m, mask

class Random_rotate(object):
    def __call__(self, rgb,m, mask):
        angle = np.random.randint(-25,25)
        h,w,_ = rgb.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(rgb, M, (w, h)), cv2.warpAffine(m, M, (w, h)), cv2.warpAffine(mask, M, (w, h))

class RandomHorizontalFlip(object):
    def __call__(self, rgb,m, mask):
        if np.random.randint(2)==1:
            rgb = rgb[:,::-1,:].copy()
            m = m[:, ::-1, :].copy()
            mask = mask[:,::-1,:].copy()
        return rgb,m, mask

class ToTensor(object):
    def __call__(self, rgb,m, mask):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)
        m = torch.from_numpy(m)
        m = m.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        return rgb,m,mask.mean(dim=0, keepdim=True)