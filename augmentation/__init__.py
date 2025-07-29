# 導入所有增強方法以便統一調用
from augmentation.brightness_contrast import adjust_brightness_contrast
from augmentation.msr_clahe import apply_msr_clahe
from augmentation.gaussian_blur import apply_gaussian_blur
from augmentation.random_erasing import random_erasing
from augmentation.rotation_scaling import random_rotation_scaling, gentle_rotation_scaling
from augmentation.crop_resize import random_crop_and_resize, gentle_crop_and_resize
from augmentation.cutmix import safe_cutmix
from augmentation.wave_aware import wave_aware_enhancement

# 增強方法字典，方便調用
AUGMENTATION_METHODS = {
    "brightness_contrast": adjust_brightness_contrast,
    "msr_clahe": apply_msr_clahe,
    "gaussian_blur": apply_gaussian_blur,
    "polygon_erase": random_erasing,
    "rotation_scaling": random_rotation_scaling,
    "gentle_rotation": gentle_rotation_scaling,
    "crop_resize": random_crop_and_resize,
    "gentle_crop": gentle_crop_and_resize,
    "cutmix": safe_cutmix,
    "wave_aware": wave_aware_enhancement
}