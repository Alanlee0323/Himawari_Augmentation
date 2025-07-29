import os
import numpy as np
from tqdm import tqdm
from utils.image_utils import load_image, save_image, has_foreground_pixels

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
        
    def copy_original_images(self, target_train_imgs_dir, target_train_masks_dir, 
                            target_val_imgs_dir, target_val_masks_dir):
        """複製原始圖像到目標目錄"""
        # 複製訓練集
        print("複製訓練集原始影像...")
        train_imgs_count = 0
        for file in tqdm(os.listdir(self.source_train_imgs_dir), desc="複製訓練集原始影像"):
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) and "_mask" not in file:
                # 複製圖像
                src_img_path = os.path.join(self.source_train_imgs_dir, file)
                file_base = os.path.splitext(file)[0]
                dst_img_path = os.path.join(target_train_imgs_dir, f"{file_base}.png")  # 統一使用PNG格式
                
                # 複製對應的遮罩
                src_mask_path = os.path.join(self.source_train_masks_dir, f"{file_base}_mask.png")
                dst_mask_path = os.path.join(target_train_masks_dir, f"{file_base}_mask.png")
                
                if os.path.exists(src_img_path) and os.path.exists(src_mask_path):
                    # 讀取和保存為PNG格式
                    img = load_image(src_img_path)
                    mask = load_image(src_mask_path)
                    
                    if img is not None and mask is not None:
                        save_image(img, dst_img_path)
                        save_image(mask, dst_mask_path)
                        train_imgs_count += 1
        
        print(f"已複製 {train_imgs_count} 個訓練集原始影像")
        
        # 複製驗證集
        print("複製驗證集原始影像...")
        val_imgs_count = 0
        for file in tqdm(os.listdir(self.source_val_imgs_dir), desc="複製驗證集原始影像"):
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) and "_mask" not in file:
                # 複製圖像
                src_img_path = os.path.join(self.source_val_imgs_dir, file)
                file_base = os.path.splitext(file)[0]
                dst_img_path = os.path.join(target_val_imgs_dir, f"{file_base}.png")  # 統一使用PNG格式
                
                # 複製對應的遮罩
                src_mask_path = os.path.join(self.source_val_masks_dir, f"{file_base}_mask.png")
                dst_mask_path = os.path.join(target_val_masks_dir, f"{file_base}_mask.png")
                
                if os.path.exists(src_img_path) and os.path.exists(src_mask_path):
                    # 讀取和保存為PNG格式
                    img = load_image(src_img_path)
                    mask = load_image(src_mask_path)
                    
                    if img is not None and mask is not None:
                        save_image(img, dst_img_path)
                        save_image(mask, dst_mask_path)
                        val_imgs_count += 1
        
        print(f"已複製 {val_imgs_count} 個驗證集原始影像")
        
        return train_imgs_count, val_imgs_count