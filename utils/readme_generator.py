import os
import datetime

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