# data/transforms.py
from PIL import Image
import torch
import torchvision.transforms as transforms

class RGBConvert:
    def __call__(self, img):
        return img if img.mode == 'RGB' else img.convert('RGB')

    def __repr__(self):
        return self.__class__.__name__

class GrayscaleConvert:
    def __call__(self, img):
        return img if img.mode == 'L' else img.convert('L')

    def __repr__(self):
        return self.__class__.__name__