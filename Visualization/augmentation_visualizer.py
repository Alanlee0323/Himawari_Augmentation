import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance, ImageFilter
import math

# 設置中文字型支援
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_rgb(path):
    """載入RGB格式的圖像"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"無法載入圖像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def adjust_brightness_contrast(image, brightness_factor=None, contrast_factor=None):
    """調整亮度與對比度，模擬不同日照條件"""
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
    return np.array(img)

def apply_msr_clahe(image):
    """應用多尺度視網膜增強(MSR)和對比度受限自適應直方圖均衡化(CLAHE)"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # 轉換為LAB色彩空間
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 應用CLAHE到亮度通道
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 多尺度視網膜增強
    sigma_list = [15, 80, 250]
    weight_list = [1/3, 1/3, 1/3]
    msr_result = np.zeros_like(cl, dtype=np.float32)
    
    for sigma, weight in zip(sigma_list, weight_list):
        blur = cv2.GaussianBlur(cl, (0, 0), sigma)
        retinex = np.log10(cl.astype(np.float32) + 1.0) - np.log10(blur.astype(np.float32) + 1.0)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        msr_result += weight * retinex
    
    msr_result = cv2.normalize(msr_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced_lab = cv2.merge([msr_result, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def apply_gaussian_blur(image, radius=None):
    """應用高斯模糊，模擬成像模糊退化"""
    if radius is None:
        radius = random.uniform(0.5, 2.0)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)

def random_erasing(img, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=None):
    """Random Erasing 數據增強方法"""
    if random.random() >= p:
        return img
    
    if r2 is None:
        r2 = 1.0 / r1
    
    h, w = img.shape[:2]
    area = h * w
    
    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)
        
        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if erase_w < w and erase_h < h:
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            
            img_out = img.copy()
            
            if len(img.shape) == 3:
                noise = np.random.randint(0, 256, (erase_h, erase_w, img.shape[2]), dtype=np.uint8)
                img_out[y:y+erase_h, x:x+erase_w, :] = noise
            else:
                noise = np.random.randint(0, 256, (erase_h, erase_w), dtype=np.uint8)
                img_out[y:y+erase_h, x:x+erase_w] = noise
            
            return img_out
    
    return img

def random_rotation_scaling(img, mask, max_angle=15, scale_range=(0.9, 1.1)):
    """應用隨機旋轉和縮放到圖像和遮罩"""
    h, w = img.shape[:2]
    
    angle = random.uniform(-max_angle, max_angle)
    scale = random.uniform(scale_range[0], scale_range[1])
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return img_rotated, mask_rotated

def random_crop_and_resize(img, mask, scale_min=0.5, scale_max=0.8):
    """隨機裁剪圖像的一部分並將其調整回原始大小"""
    h, w = img.shape[:2]
    scale = random.uniform(scale_min, scale_max)
    ch, cw = int(h * scale), int(w * scale)
    
    x = random.randint(0, h - ch) if h > ch else 0
    y = random.randint(0, w - cw) if w > cw else 0
    
    img_crop = img[x:x+ch, y:y+cw]
    mask_crop = mask[x:x+ch, y:y+cw]
    
    img_resized = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return img_resized, mask_resized

def safe_cutmix(img1, mask1, img2, mask2, alpha=3.0, max_attempts=10):
    """安全CutMix：避免切割現有內波區域"""
    for _ in range(max_attempts):
        h, w = img1.shape[:2]
        lam = np.random.beta(alpha, alpha)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        cx, cy = random.randint(0, w), random.randint(0, h)
        x1 = np.clip(cx - cut_w//2, 0, w)
        y1 = np.clip(cy - cut_h//2, 0, h)
        x2 = np.clip(cx + cut_w//2, 0, w)
        y2 = np.clip(cy + cut_h//2, 0, h)
        
        if len(mask1.shape) > 2:
            region_mask = cv2.cvtColor(mask1[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        else:
            region_mask = mask1[y1:y2, x1:x2]
            
        if np.sum(region_mask > 200) < 10:
            img_cm = img1.copy()
            mask_cm = mask1.copy()
            img_cm[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
            mask_cm[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
            return img_cm, mask_cm
    
    # 邊界混合策略
    edge_width = w // 4
    edge_height = h // 4
    
    edge_choice = random.randint(0, 3)
    img_cm = img1.copy()
    mask_cm = mask1.copy()
    
    if edge_choice == 0:
        img_cm[:edge_height, :] = img2[:edge_height, :]
        mask_cm[:edge_height, :] = mask2[:edge_height, :]
    elif edge_choice == 1:
        img_cm[:, -edge_width:] = img2[:, -edge_width:]
        mask_cm[:, -edge_width:] = mask2[:, -edge_width:]
    elif edge_choice == 2:
        img_cm[-edge_height:, :] = img2[-edge_height:, :]
        mask_cm[-edge_height:, :] = mask2[-edge_height:, :]
    else:
        img_cm[:, :edge_width] = img2[:, :edge_width]
        mask_cm[:, :edge_width] = mask2[:, :edge_width]
    
    return img_cm, mask_cm

def cutmix(img1, mask1, img2, mask2, alpha=1.0):
    """標準CutMix增強方法"""
    h, w = img1.shape[:2]
    
    lam = np.random.beta(alpha, alpha)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    img_cm = img1.copy()
    mask_cm = mask1.copy()
    
    img_cm[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    mask_cm[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    
    actual_lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
    
    return img_cm, mask_cm, actual_lam

class AugmentationVisualizer:
    """增強方法視覺化工具類"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 設置隨機種子以確保結果可重現
        random.seed(42)
        np.random.seed(42)
    
    def apply_augmentations(self, img1, mask1, img2=None, mask2=None):
        """應用所有增強方法並返回結果"""
        results = {}
        
        # 原始圖像
        results['Original'] = (img1.copy(), mask1.copy())
        
        # 1. 亮度與對比度調整
        bc_img = adjust_brightness_contrast(img1.copy(), brightness_factor=1.2, contrast_factor=1.2)
        results['Brightness & Contrast'] = (bc_img, mask1.copy())
        
        # 2. MSR + CLAHE 增強
        msr_img = apply_msr_clahe(img1.copy())
        results['MSR + CLAHE'] = (msr_img, mask1.copy())
        
        # 3. 高斯模糊
        blur_img = apply_gaussian_blur(img1.copy(), radius=1.5)
        results['Gaussian Blur'] = (blur_img, mask1.copy())
        
        # 4. 隨機擦除
        erase_img = random_erasing(img1.copy(), p=1.0)
        results['Random Erasing'] = (erase_img, mask1.copy())
        
        # 5. 旋轉與縮放
        rot_img, rot_mask = random_rotation_scaling(img1.copy(), mask1.copy(), 
                                              max_angle=15, 
                                              scale_range=(0.9, 1.1))
        results['Rotation & Scaling'] = (rot_img, rot_mask)
        
        # 6. 隨機裁剪
        crop_img, crop_mask = random_crop_and_resize(img1.copy(), mask1.copy(), 
                                               scale_min=0.5, 
                                               scale_max=0.8)
        results['Random Cropping'] = (crop_img, crop_mask)
        
        # 7. 標準 CutMix（需要第二張圖像）
        if img2 is not None and mask2 is not None:
            std_cm_img, std_cm_mask, lam = cutmix(img1.copy(), mask1.copy(), 
                                                 img2.copy(), mask2.copy(), 
                                                 alpha=1.0)
            results['CutMix'] = (std_cm_img, std_cm_mask)
        
        return results
    
    def create_all_methods_visualization(self, augmentation_results):
        """創建包含所有方法的完整視覺化"""
        titles = list(augmentation_results.keys())
        images = [augmentation_results[title][0] for title in titles]
        masks = [augmentation_results[title][1] for title in titles]
        
        n_methods = len(titles)
        
        # 創建圖表：每個方法一行，每行3列（方法名、衛星圖像、遮罩）
        fig, axes = plt.subplots(n_methods, 3, figsize=(15, 3*n_methods))
        
        # 如果只有一個方法，確保axes是2D陣列
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        # 添加列標題（只在第一行）
        if n_methods > 0:
            axes[0, 0].text(0.5, 1.1, '', ha='center', va='bottom', 
                           fontsize=18, fontweight='bold', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 1.1, 'Satellite Image', ha='center', va='bottom', 
                           fontsize=18, fontweight='bold', transform=axes[0, 1].transAxes)
            axes[0, 2].text(0.5, 1.1, 'Mask', ha='center', va='bottom', 
                           fontsize=18, fontweight='bold', transform=axes[0, 2].transAxes)
        
        for i in range(n_methods):
            # 第一列：方法名稱
            axes[i, 0].text(0.5, 0.5, titles[i], ha='center', va='center', 
                           fontsize=16, fontweight='bold', transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')
            
            # 第二列：衛星圖像
            axes[i, 1].imshow(images[i])
            axes[i, 1].axis('off')
            
            # 第三列：遮罩
            axes[i, 2].imshow(masks[i])
            axes[i, 2].axis('off')
        
        # 調整子圖間距
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.95)
        
        # 儲存為高解析度PNG
        filename = 'augmentation_all_methods.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        plt.close()
        
        print(f"完整方法視覺化已保存到: {filepath}")
        print(f"總共包含 {n_methods} 種增強方法")
        
        return filename
    
    def run_visualization(self, image1_path, mask1_path, image2_path=None, mask2_path=None):
        """執行完整的視覺化流程"""
        # 載入圖像
        img1 = load_rgb(image1_path)
        mask1 = load_rgb(mask1_path)
        
        img2 = None
        mask2 = None
        if image2_path and mask2_path:
            img2 = load_rgb(image2_path)
            mask2 = load_rgb(mask2_path)
            print("已載入第二張圖像，將包含 CutMix 方法的視覺化")
        else:
            print("未提供第二張圖像，將跳過 CutMix 方法")
        
        # 應用增強方法
        augmentation_results = self.apply_augmentations(img1, mask1, img2, mask2)
        
        # 顯示將要視覺化的方法列表
        print(f"將視覺化以下 {len(augmentation_results)} 種方法:")
        for i, method in enumerate(augmentation_results.keys(), 1):
            print(f"  {i}. {method}")
        
        # 創建包含所有方法的完整視覺化
        self.create_all_methods_visualization(augmentation_results)
        
        print("所有視覺化結果已保存到:", self.output_dir)

def main():
    """主函數"""
    base_dir = r"C:\Users\alana\Dropbox\Himawari8_Wave_Speed_Detection\Himawari8_wave_speed_detection\visualization\Vis_Augmentation"
    output_dir = os.path.join(base_dir, "Aug_Visualization_Results")
    
    # 圖像路徑
    image1_path = os.path.join(base_dir, "Aug_Visualization", "201906180340.png")
    mask1_path = os.path.join(base_dir, "Aug_Visualization", "201906180340_mask.png")
    image2_path = os.path.join(base_dir, "Aug_Visualization", "201906080720.png")
    mask2_path = os.path.join(base_dir, "Aug_Visualization", "201906080720_mask.png")
    
    # 創建視覺化工具實例
    visualizer = AugmentationVisualizer(output_dir)
    
    # 執行視覺化
    visualizer.run_visualization(image1_path, mask1_path, image2_path, mask2_path)
    print("視覺化流程完成！")

if __name__ == "__main__":
    main()