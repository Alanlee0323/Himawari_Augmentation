import numpy as np
import cv2
import random
from augmentation.brightness_contrast import adjust_brightness_contrast
from augmentation.gaussian_blur import apply_gaussian_blur
from augmentation.random_erasing import random_erasing

def wave_aware_enhancement(img, mask):
    """根據內波區域採用不同增強策略"""
    # 檢測內波區域
    if len(mask.shape) > 2:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        gray_mask = mask
    
    # 創建內波區域遮罩
    wave_mask = (gray_mask > 200).astype(np.uint8) * 255
    non_wave_mask = (gray_mask <= 200).astype(np.uint8) * 255
    
    # 對內波區域應用對比度增強和MSR+CLAHE
    enhanced_img = img.copy()
    
    # 1. 處理內波區域 - 僅增強對比度和清晰度
    wave_only = img.copy()
    wave_only[non_wave_mask == 255] = 0  # 僅保留內波區域
    
    # 內波區域增強 - 對比度提高
    wave_enhanced, _ = adjust_brightness_contrast(wave_only, brightness_factor=1.0, contrast_factor=1.3)
    
    # 2. 處理非內波區域 - 可以應用更多變形
    non_wave_only = img.copy()
    non_wave_only[wave_mask == 255] = 0  # 僅保留非內波區域
    
    # 非內波區域增強 - 隨機選擇一種增強方法
    method_choice = random.randint(0, 2)
    if method_choice == 0:
        non_wave_enhanced, _ = adjust_brightness_contrast(
            non_wave_only, 
            random.uniform(0.8, 1.2), 
            random.uniform(0.9, 1.1)
        )
    elif method_choice == 1:
        non_wave_enhanced, _ = apply_gaussian_blur(
            non_wave_only, 
            radius=random.uniform(0.5, 1.5)
        )
    else:
        non_wave_enhanced, _ = random_erasing(
            non_wave_only, 
            p=1.0,  # 確保執行擦除
            vertices=random.randint(6, 12)
        )
    
    # 合併兩個區域
    enhanced_img = cv2.add(wave_enhanced, non_wave_enhanced)
    
    return enhanced_img, mask.copy()