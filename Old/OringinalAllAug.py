import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import shutil
import datetime
import json
import math

# ================ 基礎工具函數 ================

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

def create_directory(directory):
    """創建目錄（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
def copy_file(src, dst):
    """複製文件並確保目標目錄存在"""
    dst_dir = os.path.dirname(dst)
    create_directory(dst_dir)
    shutil.copy2(src, dst)

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

# ================ 影像增強方法 ================

class ImageAugmenter:
    """影像增強類，包含各種增強方法"""
    
    @staticmethod
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

    @staticmethod
    def apply_msr_clahe(image):
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
        
        return enhanced_rgb

    @staticmethod
    def apply_gaussian_blur(image, radius=None):
        """應用高斯模糊，模擬成像模糊退化"""
        if radius is None:
            radius = random.uniform(0.5, 2.0)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 使用ImageFilter.GaussianBlur而不是Image.GaussianBlur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred)

    @staticmethod
    def random_erasing(img, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=None):
        """
        Random Erasing 數據增強方法 (Zhong et al., AAAI 2020)
        """
        if random.random() >= p:
            return img
        
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
                
                return img_out
        
        # 如果100次嘗試都無法找到合適的區域，返回原圖像
        return img

    @staticmethod
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

    @staticmethod
    def gentle_rotation_scaling(img, mask, max_angle=10, scale_range=(0.7, 1.3)):
        """溫和的旋轉與縮放，使用較小的角度和較窄的縮放範圍"""
        return ImageAugmenter.random_rotation_scaling(img, mask, max_angle=max_angle, scale_range=scale_range)

    @staticmethod
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

    @staticmethod
    def gentle_crop_and_resize(img, mask, scale_min=0.7, scale_max=0.9):
        """較溫和的裁剪，使用較大的裁剪範圍以保留更多內波結構"""
        return ImageAugmenter.random_crop_and_resize(img, mask, scale_min=scale_min, scale_max=scale_max)

    @staticmethod
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
        # 選擇圖像邊緣區域進行混合（通常邊緣不含內波）
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

    @staticmethod
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
        wave_enhanced = ImageAugmenter.adjust_brightness_contrast(wave_only, brightness_factor=1.0, contrast_factor=1.3)
        
        # 2. 處理非內波區域 - 可以應用更多變形
        non_wave_only = img.copy()
        non_wave_only[wave_mask == 255] = 0  # 僅保留非內波區域
        
        # 非內波區域增強 - 隨機選擇一種增強方法
        non_wave_methods = [
            lambda i: ImageAugmenter.adjust_brightness_contrast(i, random.uniform(0.8, 1.2), random.uniform(0.9, 1.1)),
            lambda i: ImageAugmenter.apply_gaussian_blur(i, random.uniform(0.5, 1.5)),
            lambda i: ImageAugmenter.random_erasing(i, 0.5)
        ]
        
        non_wave_enhanced = random.choice(non_wave_methods)(non_wave_only)
        
        # 合併兩個區域
        enhanced_img = cv2.add(wave_enhanced, non_wave_enhanced)
        
        return enhanced_img, mask.copy()

# ================ 實驗管理與處理 ================

class ExperimentManager:
    """實驗管理類，處理實驗配置和實驗文檔生成"""
    
    @staticmethod
    def generate_readme(experiment_dir, experiment_config):
        """生成實驗說明文件"""
        readme_path = os.path.join(experiment_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {experiment_config['name']}\n\n")
            f.write(f"生成日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 實驗說明\n\n")
            f.write(f"{experiment_config['description']}\n\n")
            
            f.write("## 增強方法比例\n\n")
            f.write("| 方法 | 比例 | 參數設置 |\n")
            f.write("|------|------|--------|\n")
            
            for method, ratio in experiment_config.get('method_ratios', {}).items():
                params = experiment_config.get('method_params', {}).get(method, "預設參數")
                if isinstance(params, dict):
                    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                else:
                    params_str = str(params)
                f.write(f"| {method} | {ratio*100:.1f}% | {params_str} |\n")
            
            f.write("\n## 數據集統計\n\n")
            f.write(f"- 原始訓練集內波影像數量: {experiment_config.get('original_train_count', 'N/A')}\n")
            f.write(f"- 原始驗證集內波影像數量: {experiment_config.get('original_val_count', 'N/A')}\n")
            f.write(f"- 擴增後訓練集總數量: {experiment_config.get('total_train_count', 'N/A')}\n")
            f.write(f"- 擴增後驗證集總數量: {experiment_config.get('total_val_count', 'N/A')}\n")
            f.write(f"- 訓練集擴增倍數: {experiment_config.get('train_augmentation_factor', 'N/A')}\n")
            f.write(f"- 驗證集擴增倍數: {experiment_config.get('val_augmentation_factor', 'N/A')}\n")
            
            if 'train_method_counts' in experiment_config:
                f.write("\n## 訓練集各方法實際擴增數量\n\n")
                for method, count in experiment_config['train_method_counts'].items():
                    f.write(f"- {method}: {count} 張\n")
                    
            if 'val_method_counts' in experiment_config:
                f.write("\n## 驗證集各方法實際擴增數量\n\n")
                for method, count in experiment_config['val_method_counts'].items():
                    f.write(f"- {method}: {count} 張\n")

# ================ 主要數據處理模塊 ================

class DataProcessor:
    """數據處理類，處理數據掃描、加載和準備"""
    
    def __init__(self, source_data_dir):
        """初始化數據處理器"""
        self.source_data_dir = source_data_dir
        self.source_train_imgs_dir = os.path.join(source_data_dir, "train", "imgs")
        self.source_train_masks_dir = os.path.join(source_data_dir, "train", "masks")
        self.source_val_imgs_dir = os.path.join(source_data_dir, "val", "imgs")
        self.source_val_masks_dir = os.path.join(source_data_dir, "val", "masks")
        
        # 統計數據
        self.train_isw_file_bases = []
        self.val_isw_file_bases = []
        self.train_isw_images = {}
        self.train_isw_masks = {}
        self.val_isw_images = {}
        self.val_isw_masks = {}
        
    def scan_dataset(self):
        """掃描數據集，找出含有內波的影像"""
        # 掃描訓練集
        print("掃描訓練集中含有ISW的影像...")
        train_mask_files = [f for f in os.listdir(self.source_train_masks_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for mask_file in tqdm(train_mask_files, desc="檢查訓練集內波遮罩"):
            mask_path = os.path.join(self.source_train_masks_dir, mask_file)
            if has_foreground_pixels(mask_path):
                file_base = mask_file.replace("_mask", "")
                file_base = os.path.splitext(file_base)[0]
                self.train_isw_file_bases.append(file_base)
        
        print(f"找到 {len(self.train_isw_file_bases)} 個訓練集中含有ISW的影像檔案")
        
        # 掃描驗證集
        print("掃描驗證集中含有ISW的影像...")
        val_mask_files = [f for f in os.listdir(self.source_val_masks_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for mask_file in tqdm(val_mask_files, desc="檢查驗證集內波遮罩"):
            mask_path = os.path.join(self.source_val_masks_dir, mask_file)
            if has_foreground_pixels(mask_path):
                file_base = mask_file.replace("_mask", "")
                file_base = os.path.splitext(file_base)[0]
                self.val_isw_file_bases.append(file_base)
        
        print(f"找到 {len(self.val_isw_file_bases)} 個驗證集中含有ISW的影像檔案")
        
    def load_images_to_memory(self):
        """將影像加載到記憶體以加速處理"""
        # 加載訓練集影像
        print("讀取訓練集ISW影像到記憶體...")
        for file_base in tqdm(self.train_isw_file_bases, desc="讀取訓練集ISW影像"):
            img_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = os.path.join(self.source_train_imgs_dir, f"{file_base}{ext}")
                if os.path.exists(img_path):
                    mask_path = os.path.join(self.source_train_masks_dir, f"{file_base}_mask.png")
                    if os.path.exists(mask_path):
                        img = load_image(img_path)
                        mask = load_image(mask_path)
                        if img is not None and mask is not None:
                            self.train_isw_images[file_base] = img
                            self.train_isw_masks[file_base] = mask
                            img_found = True
                            break
            
            if not img_found:
                print(f"警告: 找不到訓練集影像或遮罩檔案: {file_base}")
        
        print(f"成功讀取 {len(self.train_isw_images)} 個訓練集ISW影像")
        
        # 加載驗證集影像
        print("讀取驗證集ISW影像到記憶體...")
        for file_base in tqdm(self.val_isw_file_bases, desc="讀取驗證集ISW影像"):
            img_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = os.path.join(self.source_val_imgs_dir, f"{file_base}{ext}")
                if os.path.exists(img_path):
                    mask_path = os.path.join(self.source_val_masks_dir, f"{file_base}_mask.png")
                    if os.path.exists(mask_path):
                        img = load_image(img_path)
                        mask = load_image(mask_path)
                        if img is not None and mask is not None:
                            self.val_isw_images[file_base] = img
                            self.val_isw_masks[file_base] = mask
                            img_found = True
                            break
            
            if not img_found:
                print(f"警告: 找不到驗證集影像或遮罩檔案: {file_base}")
        
        print(f"成功讀取 {len(self.val_isw_images)} 個驗證集ISW影像")

    def get_stats(self):
        """取得數據集統計信息"""
        return {
            'original_train_count': len(self.train_isw_file_bases),
            'original_val_count': len(self.val_isw_file_bases),
            'train_files_count': len(self.train_isw_images),
            'val_files_count': len(self.val_isw_images)
        }

# ================ 實驗增強處理模塊 ================

