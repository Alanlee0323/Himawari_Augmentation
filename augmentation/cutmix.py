import numpy as np
import cv2
import random

def safe_cutmix(img1, mask1, img2, mask2, alpha=3.0, max_attempts=10):
    """安全CutMix：避免切割現有內波區域"""
    for _ in range(max_attempts):
        # 正常CutMix處理前先計算切割區域
        h, w = img1.shape[:2]
        lam = np.random.beta(alpha, alpha)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        cx, cy = random.randint(0, w), random.randint(0, h)
        x1 = np.clip(cx - cut_w//2, 0, w)
        y1 = np.clip(cy - cut_h//2, 0, h)
        x2 = np.clip(cx + cut_w//2, 0, w)
        y2 = np.clip(cy + cut_h//2, 0, h)
        
        # 檢查區域內是否有內波（白色像素）
        # 簡單處理：轉為灰度後檢查亮度值
        if len(mask1.shape) > 2:
            region_mask = cv2.cvtColor(mask1[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        else:
            region_mask = mask1[y1:y2, x1:x2]
                
        if np.sum(region_mask > 200) < 10:  # 假設少於10個白色像素表示無內波
            # 安全區域，可以進行混合
            img_cm = img1.copy()
            mask_cm = mask1.copy()
            img_cm[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
            mask_cm[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
            return img_cm, mask_cm
    
    # 若多次嘗試都無法找到合適區域，使用邊界混合的方式
    edge_width = w // 4
    edge_height = h // 4
    
    # 隨機選擇一個邊緣區域
    edge_choice = random.randint(0, 3)
    if edge_choice == 0:  # 上邊緣
        img_cm = img1.copy()
        mask_cm = mask1.copy()
        img_cm[:edge_height, :] = img2[:edge_height, :]
        mask_cm[:edge_height, :] = mask2[:edge_height, :]
    elif edge_choice == 1:  # 右邊緣
        img_cm = img1.copy()
        mask_cm = mask1.copy()
        img_cm[:, -edge_width:] = img2[:, -edge_width:]
        mask_cm[:, -edge_width:] = mask2[:, -edge_width:]
    elif edge_choice == 2:  # 下邊緣
        img_cm = img1.copy()
        mask_cm = mask1.copy()
        img_cm[-edge_height:, :] = img2[-edge_height:, :]
        mask_cm[-edge_height:, :] = mask2[-edge_height:, :]
    else:  # 左邊緣
        img_cm = img1.copy()
        mask_cm = mask1.copy()
        img_cm[:, :edge_width] = img2[:, :edge_width]
        mask_cm[:, :edge_width] = mask2[:, :edge_width]
    
    return img_cm, mask_cm