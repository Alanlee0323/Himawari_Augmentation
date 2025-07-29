import numpy as np
import cv2

def apply_msr_clahe(image, mask=None):
    """應用多尺度視網膜增強(MSR)和對比度受限自適應直方圖均衡化(CLAHE)"""
    # 確保輸入是OpenCV格式
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # 轉換為LAB色彩空間（更適合CLAHE處理）
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 應用CLAHE到亮度通道
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 多尺度視網膜增強 (簡化版)
    # 使用三個高斯核進行視網膜增強
    sigma_list = [15, 80, 250]
    weight_list = [1/3, 1/3, 1/3]  # 均衡權重
    msr_result = np.zeros_like(cl, dtype=np.float32)
    
    for sigma, weight in zip(sigma_list, weight_list):
        # 對亮度通道應用高斯模糊
        blur = cv2.GaussianBlur(cl, (0, 0), sigma)
        # 計算視網膜響應 (對數域中的差異)
        retinex = np.log10(cl.astype(np.float32) + 1.0) - np.log10(blur.astype(np.float32) + 1.0)
        # 正規化到0-255
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        # 累積權重結果
        msr_result += weight * retinex
    
    # 將結果轉換回8位無符號整數
    msr_result = cv2.normalize(msr_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 將處理後的亮度通道與原始a、b通道合併
    enhanced_lab = cv2.merge([msr_result, a, b])
    # 轉換回RGB色彩空間
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 遮罩不變
    return enhanced_rgb, mask.copy() if mask is not None else None