import numpy as np
import random
from PIL import Image, ImageEnhance

def adjust_brightness_contrast(image, mask=None, brightness_factor=None, contrast_factor=None):
    """調整亮度與對比度，模擬不同日照條件
    
    參數:
        image: 輸入圖像
        mask: 輸入遮罩（可選）
        brightness_factor: 亮度調整因子（可選，默認隨機）
        contrast_factor: 對比度調整因子（可選，默認隨機）
        contrast_factor: 對比度調整因子（可選，默認隨機）
        
    返回:
        處理後的圖像和原始遮罩
    """
    try:
        if brightness_factor is None:
            brightness_factor = random.uniform(0.8, 1.2)
        if contrast_factor is None:
            contrast_factor = random.uniform(0.8, 1.2)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img = image
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        return np.array(img), mask
    
    except Exception as e:
        print(f"亮度對比度調整出錯: {e}")
        return None, None