def get_experiment_config():
    """獲取消融實驗配置"""
    return {
        "Experiment_A": {
            "name": "實驗A: 僅亮度對比度增強",
            "description": "此實驗僅使用亮度對比度調整。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 1.0
            }
        },
        "Experiment_B": {
            "name": "實驗B: 僅msr_clahe增強",
            "description": "此實驗僅使用msr_clahe調整。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "msr_clahe": 1.0            # 50%
            }
        },
        "Experiment_C": {
            "name": "實驗C: 僅高斯模糊增強",
            "description": "此實驗僅使用高斯模糊。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
               "gaussian_blur": 1.0            # 100%
            }
        },
        "Experiment_D": {
            "name": "實驗D: 僅亮度與對比度增強",
            "description": "此實驗僅使用亮度對比度調整和MSR+CLAHE增強，不進行任何幾何變換。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 0.5,  # 50%
                "msr_clahe": 0.5            # 50%
            }
        },
        "Experiment_E": {
            "name": "實驗E: msr_clahe+高斯模糊",
            "description": "此實驗使用msr_clahe+高斯模糊。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "msr_clahe": 0.8,           # 80%
                "gaussian_blur": 0.2        # 20%
            }   
        },

        "Experiment_F": {
            "name": "實驗F: 亮度對比度與多尺度紋理增強與模糊",
            "description": "此實驗在亮度對比度與多尺度紋理增強的基礎上添加高斯模糊，評估模糊對模型性能的影響。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 0.4,  # 40%
                "msr_clahe": 0.4,           # 40%
                "gaussian_blur": 0.2        # 20%
            }
        },

        "Experiment_G": {
            "name": "實驗G: 多尺度紋理增強+亮度對比度調整+雜訊模擬增強+遮罩模擬增強",
            "description": "此實驗在多尺度紋理增強+亮度對比度調整+雜訊模擬增強的基礎上添加隨機擦除，評估擦除對模型處理部分遮擋的能力影響。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 0.35,  # 35%
                "msr_clahe": 0.35,           # 35%
                "gaussian_blur": 0.15,       # 15%
                "polygon_erase": 0.15        # 15%
            },
            "method_params": {
                "polygon_erase": {
                    "p": 0.5,             # 應用概率
                    "sl": 0.02,           # 最小擦除面積比例
                    "sh": 0.2,            # 最大擦除面積比例
                    "r1": 0.3,            # 最小寬高比
                    "vertices": 8         # 多邊形頂點數
                },
                "gaussian_blur": {
                    "radius_min": 0.5,
                    "radius_max": 1.5     # 稍微降低最大模糊半徑
                }
            }
        },

        "Experiment_H": {
            "name": "實驗H: 完整擴增組合",
            "description": "此實驗結合所有擴增方法，包括多尺度紋理增強+亮度對比度調整+雜訊模擬增強+遮罩模擬增強+幾何變換增強和CutMix。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 0.25,  # 25%
                "msr_clahe": 0.25,           # 25%
                "gaussian_blur": 0.15,       # 15%
                "gentle_rotation": 0.15,     # 15%
                "gentle_crop": 0.15,         # 15%
                "cutmix": 0.05               # 5%
            },

            "method_params": {
                "gentle_rotation": {
                    "max_angle": 10,
                    "scale_min": 0.95,
                    "scale_max": 1.05
                },
                "gentle_crop": {
                    "scale_min": 0.7,
                    "scale_max": 0.9
                }
            }
        },
        
        "Experiment_I": {
            "name": "實驗I: 溫和幾何變換",
            "description": "此實驗在多尺度紋理增強+亮度對比度調整+雜訊模擬增強基礎上添加溫和的幾何變換，包括溫和旋轉和溫和裁剪。",
            "augmentation_factor": 2.0,
            "val_augmentation_factor": 1.0,
            "method_ratios": {
                "brightness_contrast": 0.3,  # 30%
                "msr_clahe": 0.3,           # 30%
                "gaussian_blur": 0.15,      # 15%
                "gentle_rotation": 0.125,   # 12.5%
                "gentle_crop": 0.125        # 12.5%
            },
            "method_params": {
                "gentle_rotation": {
                    "max_angle": 10,
                    "scale_min": 0.95,
                    "scale_max": 1.05
                },
                "gentle_crop": {
                    "scale_min": 0.7,
                    "scale_max": 0.9
                }
            }
        },
    }