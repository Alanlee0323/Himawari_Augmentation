import numpy as np
import cv2
import random

def random_rotation_scaling(img, mask, max_angle=15, scale_range=(0.9, 1.1)):
    """Apply random rotation and scaling to both image and mask."""
    h, w = img.shape[:2]
    
    # 隨機角度和縮放比例
    angle = random.uniform(-max_angle, max_angle)
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # 計算旋轉矩陣
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 應用變換到圖像和遮罩
    img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return img_rotated, mask_rotated

def gentle_rotation_scaling(img, mask, max_angle=10, scale_range=(0.7, 1.3)):
    """溫和的旋轉與縮放，使用較小的角度和較窄的縮放範圍"""
    return random_rotation_scaling(img, mask, max_angle=max_angle, scale_range=scale_range)