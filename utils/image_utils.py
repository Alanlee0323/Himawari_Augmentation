import numpy as np
import cv2
from PIL import Image

def load_image(image_path):
    """讀取圖像並轉換為numpy數組"""
    try:
        image = Image.open(image_path)
        return np.array(image)
    except Exception as e:
        print(f"錯誤: 無法讀取影像 {image_path}: {e}")
        return None

def save_image(image, path):
    """保存numpy數組為圖像"""
    try:
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(path)
        else:
            image.save(path)
    except Exception as e:
        print(f"錯誤: 無法保存影像至 {path}: {e}")

def has_foreground_pixels(mask_path, threshold=5):
    """檢查mask是否含有前景像素（白色區域）"""
    try:
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) > 2:  # 如果是RGB圖像
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        # 計算白色像素數量（值為255的像素）
        white_pixels = np.sum(mask == 255)
        return white_pixels > threshold  # 如果白色像素超過閾值，則認為有前景
    except Exception as e:
        print(f"錯誤: 處理遮罩 {mask_path} 時發生錯誤: {e}")
        return False