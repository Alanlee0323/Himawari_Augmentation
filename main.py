import os
import random
import numpy as np
from tqdm import tqdm

from utils.file_utils import create_directory
from utils.image_utils import save_image
from utils.readme_generator import generate_readme
from augmentation import AUGMENTATION_METHODS
from data_processor import DataProcessor
from config import get_experiment_config

class AugmentationExperimentRunner:
    """實驗運行管理類"""
    
    def __init__(self, source_data_dir, output_base_dir):
        """初始化實驗運行器"""
        self.source_data_dir = source_data_dir
        self.output_base_dir = output_base_dir
        create_directory(output_base_dir)
        
        # 初始化數據處理器
        self.data_processor = DataProcessor(source_data_dir)
        
    def run_experiments(self, experiments_config):
        """執行所有實驗"""
        # 掃描和加載數據
        self.data_processor.scan_dataset()
        
        stats = self.data_processor.get_stats()
        if stats['original_train_count'] == 0 and stats['original_val_count'] == 0:
            print("未找到含有ISW的影像，無法進行擴增")
            return
        
        self.data_processor.load_images_to_memory()
        
        # 對每個實驗配置執行增強
        for experiment_name, experiment_config in experiments_config.items():
            self.run_single_experiment(experiment_name, experiment_config)
    
    def run_single_experiment(self, experiment_name, experiment_config):
        """執行單個實驗"""
        print(f"\n開始執行實驗: {experiment_name}")
        
        # 創建實驗目錄
        experiment_dir = os.path.join(self.output_base_dir, experiment_name)
        create_directory(experiment_dir)
        
        # 創建實驗的train和val目錄及子目錄
        experiment_train_dir = os.path.join(experiment_dir, "train")
        experiment_train_imgs_dir = os.path.join(experiment_train_dir, "imgs")
        experiment_train_masks_dir = os.path.join(experiment_train_dir, "masks")
        
        experiment_val_dir = os.path.join(experiment_dir, "val")
        experiment_val_imgs_dir = os.path.join(experiment_val_dir, "imgs")
        experiment_val_masks_dir = os.path.join(experiment_val_dir, "masks")
        
        # 創建所有所需目錄
        for directory in [experiment_train_dir, experiment_train_imgs_dir, experiment_train_masks_dir,
                         experiment_val_dir, experiment_val_imgs_dir, experiment_val_masks_dir]:
            create_directory(directory)
        
        # 複製原始圖像
        train_imgs_count, val_imgs_count = self.data_processor.copy_original_images(
            experiment_train_imgs_dir, experiment_train_masks_dir,
            experiment_val_imgs_dir, experiment_val_masks_dir
        )
        
        # 獲取統計信息
        stats = self.data_processor.get_stats()
        original_train_count = stats['original_train_count']
        original_val_count = stats['original_val_count']
        
        # 執行訓練集增強
        train_augmentation_count = 0
        train_method_trackers = {}
        
        # 處理訓練集增強
        if experiment_config.get('method_ratios'):
            augmentation_factor = experiment_config.get('augmentation_factor', 1.0)
            train_total_target = int(original_train_count * augmentation_factor)
            train_augmentations_needed = train_total_target - original_train_count
            
            if train_augmentations_needed > 0:
                train_augmentation_count, train_method_trackers = self._augment_dataset(
                    experiment_config,
                    self.data_processor.train_isw_file_bases,
                    self.data_processor.train_isw_images,
                    self.data_processor.train_isw_masks,
                    experiment_train_imgs_dir,
                    experiment_train_masks_dir,
                    "訓練集",
                    train_augmentations_needed
                )
        
        # 執行驗證集增強
        val_augmentation_count = 0
        val_method_trackers = {}
        
        # 處理驗證集增強
        if experiment_config.get('method_ratios'):
            val_augmentation_factor = experiment_config.get('val_augmentation_factor', 
                                                         experiment_config.get('augmentation_factor', 1.0))
            val_total_target = int(original_val_count * val_augmentation_factor)
            val_augmentations_needed = val_total_target - original_val_count
            
            if val_augmentations_needed > 0:
                val_augmentation_count, val_method_trackers = self._augment_dataset(
                    experiment_config,
                    self.data_processor.val_isw_file_bases,
                    self.data_processor.val_isw_images,
                    self.data_processor.val_isw_masks,
                    experiment_val_imgs_dir,
                    experiment_val_masks_dir,
                    "驗證集",
                    val_augmentations_needed
                )
        
        # 更新實驗配置以包含實際擴增數量
        experiment_config.update({
            'original_train_count': original_train_count,
            'original_val_count': original_val_count,
            'total_train_count': train_imgs_count + train_augmentation_count,
            'total_val_count': val_imgs_count + val_augmentation_count,
            'train_augmentation_factor': (train_imgs_count + train_augmentation_count) / train_imgs_count if train_imgs_count > 0 else 0,
            'val_augmentation_factor': (val_imgs_count + val_augmentation_count) / val_imgs_count if val_imgs_count > 0 else 0,
            'train_method_counts': train_method_trackers,
            'val_method_counts': val_method_trackers
        })
        
        # 生成README
        generate_readme(experiment_dir, experiment_config)
        
        print(f"實驗 {experiment_name} 完成!")
        print(f"訓練集總數據集大小: {train_imgs_count + train_augmentation_count} 張 (原始: {train_imgs_count}, 擴增: {train_augmentation_count})")
        print(f"驗證集總數據集大小: {val_imgs_count + val_augmentation_count} 張 (原始: {val_imgs_count}, 擴增: {val_augmentation_count})")
    
    def _augment_dataset(self, experiment_config, file_bases, images, masks, 
                      target_imgs_dir, target_masks_dir, dataset_name, augmentations_needed):
        """增強數據集"""
        # 按照比例分配各方法的數量
        method_ratios = experiment_config.get('method_ratios', {})
        method_counts = {method: int(augmentations_needed * ratio) for method, ratio in method_ratios.items()}
        
        # 調整數量確保總數正確
        total_allocated = sum(method_counts.values())
        if total_allocated < augmentations_needed:
            # 將剩餘數量添加到第一個方法
            first_method = list(method_counts.keys())[0]
            method_counts[first_method] += (augmentations_needed - total_allocated)
        
        print(f"{dataset_name}各擴增方法的分配數量:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} 張")
        
        print(f"{dataset_name}總擴增數量: {sum(method_counts.values())} 張")
        
        # 為每個方法創建一個追蹤計數器
        method_trackers = {method: 0 for method in method_counts.keys()}
        
        # 按照每個方法的分配數量進行擴增
        augmentation_count = 0
        augmentation_idx = 0
        files_count = len(images)
        
        for method in method_counts.keys():
            target_count = method_counts[method]
            print(f"開始處理{dataset_name} {method} 擴增 (目標: {target_count} 張)...")
            
            # 平均分配給每個文件
            while method_trackers[method] < target_count:
                # 循環使用文件
                file_base = file_bases[augmentation_idx % files_count]
                augmentation_idx += 1
                
                if file_base not in images or file_base not in masks:
                    continue
                
                img = images[file_base]
                mask = masks[file_base]
                
                # 應用增強方法
                aug_img, aug_mask = self._apply_augmentation(method, img, mask, experiment_config, 
                                                          images, masks, file_base)
                
                if aug_img is None or aug_mask is None:
                    print(f"警告: {method} 增強方法未能產生有效結果")
                    continue
                
                # 生成新的檔案名稱
                counter = method_trackers[method] + 1
                new_img_name = f"{file_base}_aug_{method}_{counter:03d}.png"
                new_mask_name = f"{file_base}_aug_{method}_{counter:03d}_mask.png"
                
                # 保存擴增圖像
                save_image(aug_img, os.path.join(target_imgs_dir, new_img_name))
                save_image(aug_mask, os.path.join(target_masks_dir, new_mask_name))
                
                method_trackers[method] += 1
                augmentation_count += 1
                
                # 提供進度更新
                if augmentation_count % 100 == 0:
                    print(f"已完成 {augmentation_count} / {augmentations_needed} {dataset_name}擴增")
        
        return augmentation_count, method_trackers
    
    def _apply_augmentation(self, method, img, mask, experiment_config, images, masks, file_base):
        """應用指定的增強方法"""
        method_params = experiment_config.get('method_params', {}).get(method, {})
        
        # CutMix 特殊處理，需要兩張圖像
        if method == "cutmix":
            # 隨機選擇另一張圖像
            other_file_base = random.choice(list(images.keys()))
            while other_file_base == file_base and len(images) > 1:
                other_file_base = random.choice(list(images.keys()))
            
            other_img = images[other_file_base]
            other_mask = masks[other_file_base]
            
            alpha = method_params.get('alpha', 3.0)
            max_attempts = method_params.get('max_attempts', 10)
            
            aug_func = AUGMENTATION_METHODS[method]
            return aug_func(img, mask, other_img, other_mask, alpha=alpha, max_attempts=max_attempts)
        
        # 其他方法的處理
        if method in AUGMENTATION_METHODS:
            aug_func = AUGMENTATION_METHODS[method]
            
            # 根據不同方法處理不同參數
            if method == "brightness_contrast":
                brightness_min = method_params.get('brightness_min', 0.7)
                brightness_max = method_params.get('brightness_max', 1.3)
                contrast_min = method_params.get('contrast_min', 0.8)
                contrast_max = method_params.get('contrast_max', 1.2)
                brightness_factor = random.uniform(brightness_min, brightness_max)
                contrast_factor = random.uniform(contrast_min, contrast_max)
                return aug_func(img, mask, brightness_factor=brightness_factor, contrast_factor=contrast_factor)
                
            elif method == "gaussian_blur":
                radius_min = method_params.get('radius_min', 0.5)
                radius_max = method_params.get('radius_max', 2.0)
                radius = random.uniform(radius_min, radius_max)
                return aug_func(img, mask, radius=radius)
            
            elif method == "polygon_erase":
                vertices_min = method_params.get('vertices_min', 6)
                vertices_max = method_params.get('vertices_max', 12)
                vertices = random.randint(vertices_min, vertices_max)
                return aug_func(img, mask, p=1.0, vertices=vertices)
            
            elif method in ["rotation_scaling", "gentle_rotation"]:
                max_angle = method_params.get('max_angle', 15 if method == "rotation_scaling" else 10)
                scale_min = method_params.get('scale_min', 0.9 if method == "rotation_scaling" else 0.95)
                scale_max = method_params.get('scale_max', 1.1 if method == "rotation_scaling" else 1.05)
                scale_range = (scale_min, scale_max)
                return aug_func(img, mask, max_angle=max_angle, scale_range=scale_range)
            
            elif method in ["crop_resize", "gentle_crop"]:
                scale_min = method_params.get('scale_min', 0.5 if method == "crop_resize" else 0.7)
                scale_max = method_params.get('scale_max', 0.8 if method == "crop_resize" else 0.9)
                return aug_func(img, mask, scale_min=scale_min, scale_max=scale_max)
            
            else:
                # 其他方法直接調用
                return aug_func(img, mask)
        else:
            print(f"未知增強方法: {method}")
            return None, None

def main():
    # 設定基礎目錄
    base_data_dir = r"C:\Users\alana\Dropbox\Himawari8_Wave_Speed_Detection\Himawari8_wave_speed_detection\data"
    source_data_dir = os.path.join(base_data_dir, "data")  # 包含train和val子目錄的源數據目錄
    
    # 設定消融實驗輸出目錄（直接在base_data_dir下創建實驗目錄）
    output_base_dir = base_data_dir
    
    # 初始化實驗運行器
    experiment_runner = AugmentationExperimentRunner(source_data_dir, output_base_dir)
    
    # 獲取實驗配置並運行實驗
    experiments_config = get_experiment_config()
    experiment_runner.run_experiments(experiments_config)

if __name__ == "__main__":
    main()