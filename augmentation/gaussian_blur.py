import numpy as np
import random
from PIL import Image, ImageFilter

def apply_gaussian_blur(image, mask=None, radius=None):
    """應用高斯模糊，模擬成像模糊退化"""
    if radius is None:
        radius = random.uniform(0.5, 2.0)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 使用ImageFilter.GaussianBlur
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # 遮罩不變
    return np.array(blurred), mask.copy() if mask is not None else None