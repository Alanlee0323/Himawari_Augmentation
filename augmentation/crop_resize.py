import numpy as np
import cv2
import random

def random_crop_and_resize(img, mask, scale_min=0.5, scale_max=0.8):
    """隨機裁剪圖像的一部分並將其調整回原始大小 (同時處理image和mask)"""
    h, w = img.shape[:2]
    scale = random.uniform(scale_min, scale_max)
    ch, cw = int(h * scale), int(w * scale)
    
    # 確保裁剪區域不超出圖像邊界
    x = random.randint(0, h - ch) if h > ch else 0
    y = random.randint(0, w - cw) if w > cw else 0
    
    # 裁剪圖像和遮罩
    img_crop = img[x:x+ch, y:y+cw]
    mask_crop = mask[x:x+ch, y:y+cw]
    
    # 將裁剪的區域調整回原始大小
    img_resized = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 使用最近鄰插值以保持mask的類別標籤
    mask_resized = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return img_resized, mask_resized

def gentle_crop_and_resize(img, mask, scale_min=0.7, scale_max=0.9):
    """較溫和的裁剪，使用較大的裁剪範圍以保留更多內波結構"""
    return random_crop_and_resize(img, mask, scale_min=scale_min, scale_max=scale_max)