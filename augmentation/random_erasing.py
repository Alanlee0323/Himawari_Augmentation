import numpy as np
import random
import math

def random_erasing(img, mask=None, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=None, vertices=None):
    """
    Random Erasing 數據增強方法 (Zhong et al., AAAI 2020)
    """
    if random.random() >= p:
        return img, mask.copy() if mask is not None else None
    
    if r2 is None:
        r2 = 1.0 / r1  # 論文中設置r2 = 1/r1
    
    h, w = img.shape[:2]
    area = h * w
    
    # 嘗試最多100次找到合適的擦除區域
    for _ in range(100):
        # 隨機選擇擦除區域的面積和寬高比
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)
        
        # 計算擦除區域的寬和高
        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        # 確保擦除區域在圖像範圍內
        if erase_w < w and erase_h < h:
            # 隨機選擇擦除區域的左上角坐標
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            
            # 創建一個與原圖像相同形狀的副本
            img_out = img.copy()
            
            # 將擦除區域填充為隨機值
            if len(img.shape) == 3:  # 彩色圖像
                # 論文中使用的RE-R方式：每個像素使用隨機值
                noise = np.random.randint(0, 256, (erase_h, erase_w, img.shape[2]), dtype=np.uint8)
                img_out[y:y+erase_h, x:x+erase_w, :] = noise
            else:  # 灰階圖像
                noise = np.random.randint(0, 256, (erase_h, erase_w), dtype=np.uint8)
                img_out[y:y+erase_h, x:x+erase_w] = noise
            
            return img_out, mask.copy() if mask is not None else None
    
    # 如果100次嘗試都無法找到合適的區域，返回原圖像
    return img, mask.copy() if mask is not None else None