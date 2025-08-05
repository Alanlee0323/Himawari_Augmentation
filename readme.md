# 內波影像分割之數據增強消融實驗框架

> Data Augmentation Ablation Study Framework for Internal Wave Image Segmentation

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

這是一個專為海洋內波 (Internal Wave, ISW) 影像分割任務設計的數據增強 (Data Augmentation) 框架。本專案旨在系統化地執行、管理並評估多種數據增強技術對模型性能的影響。透過靈活的設定檔，使用者可以輕鬆組合不同的增強方法、調整參數，並為每次實驗自動生成詳細的說明文件。

## 專案目標 🎯

深度學習模型在影像分割任務上的成功，高度依賴於大量且多樣化的訓練數據。在海洋內波偵測領域，獲取帶有精確標註的影像成本高昂。本專案的目標是：

- **自動化數據增強**：建立一個自動化流程，根據預先定義的策略，對現有的內波影像數據集進行擴增
- **系統化消融實驗**：透過設定檔輕鬆定義一系列的消融實驗 (Ablation Study)，用以評估單一或組合增強方法對模型訓練的具體影響
- **可重現性與文檔化**：為每一次實驗自動生成一份 `README.md` 報告，詳細記錄該次實驗所使用的增強方法、參數、數據集統計等資訊，確保實驗的可重現性

## 核心功能 ✨

- **配置驅動的實驗**：所有實驗皆由 `config.py` 檔案定義。使用者無需修改核心程式碼，僅需在設定檔中增加或修改實驗配置，即可設計新的增強方案
- **豐富的增強方法庫**：內建多種針對遙測影像設計的增強方法，涵蓋像素級、幾何級以及更複雜的混合策略
- **智慧數據處理**：`DataProcessor` 模組能自動掃描數據集，僅針對包含內波前景的影像進行增強，提高處理效率
- **自動化報告生成**：實驗完成後，`readme_generator.py` 會自動在每個實驗的資料夾內，根據其實驗配置生成一份詳細的 Markdown 格式報告

## 專案結構 📂

```
.
├── augmentation/               # 數據增強方法模組
│   ├── brightness_contrast.py    # 亮度與對比度調整
│   ├── crop_resize.py          # 隨機裁切與縮放
│   ├── cutmix.py               # CutMix
│   ├── gaussian_blur.py        # 高斯模糊
│   ├── msr_clahe.py            # MSR 與 CLAHE 增強
│   ├── random_erasing.py       # 隨機擦除
│   ├── rotation_scaling.py     # 隨機旋轉與縮放
│   ├── wave_aware.py           # 感知內波區域的增強
│   └── __init__.py             # 統一匯出所有增強方法
├── utils/                      # 工具函式模組
│   ├── file_utils.py           # 檔案操作工具
│   ├── image_utils.py          # 影像讀寫與處理工具
│   └── readme_generator.py     # README 報告生成器
├── config.py                   # 實驗設定檔
├── data_processor.py           # 數據預處理與掃描器
├── main.py                     # 主程式進入點
└── README.md                   # 專案說明文件
```

## 如何使用 🚀

### 步驟一：準備數據

1. **準備您的原始數據集**，並依照以下結構存放：

```
<您的數據根目錄>/
└── data/
    ├── train/
    │   ├── imgs/       # 訓練影像
    │   └── masks/      # 訓練遮罩
    └── val/
        ├── imgs/       # 驗證影像
        └── masks/      # 驗證遮罩
```

2. **遮罩格式要求**：遮罩 (mask) 檔案應為二值化影像，其中內波區域為白色 (像素值 255)，背景為黑色 (像素值 0)

### 步驟二：定義實驗

1. 開啟 `config.py` 檔案
2. 在 `get_experiment_config()` 函數返回的字典中，您可以修改現有的實驗（如 "Experiment_A"）或新增一個
3. **實驗配置範例**：

```python
"Experiment_X": {
    "name": "實驗X: 我的自訂增強",
    "description": "描述此實驗的目的與方法。",
    "augmentation_factor": 3.0,  # 訓練集擴增倍數 (3.0代表擴增後總數為原始的3倍)
    "val_augmentation_factor": 1.0, # 驗證集擴增倍數 (1.0代表不擴增)
    "method_ratios": {
        "brightness_contrast": 0.4,  # 40% 的擴增影像使用此方法
        "msr_clahe": 0.4,           # 40% 的擴增影像使用此方法
        "gentle_rotation": 0.2        # 20% 的擴增影像使用此方法
    },
    "method_params": {
        "gentle_rotation": {         # 為特定方法設定客製化參數
            "max_angle": 5,
            "scale_min": 0.98,
            "scale_max": 1.02
        }
    }
}
```

### 步驟三：運行實驗

1. 開啟 `main.py` 檔案
2. 修改 `main` 函數中的 `base_data_dir` 路徑，使其指向您在**步驟一**中設定的 `<您的數據根目錄>`：

```python
def main():
    # 設定基礎目錄
    base_data_dir = r"C:\path\to\your\data\root"  # <--- 修改此路徑
    source_data_dir = os.path.join(base_data_dir, "data")
    output_base_dir = base_data_dir
    # ...
```

3. **執行主程式**：

```bash
python main.py
```

### 步驟四：查看結果

程式運行結束後，會在您的 `<您的數據根目錄>` 下，為 `config.py` 中定義的每一個實驗建立一個對應的資料夾（例如 `Experiment_A`, `Experiment_B` 等）。

每個實驗資料夾內都包含：
- 擴增後的 `train` 和 `val` 數據集
- 一份詳細的 `README.md` 報告，說明該次實驗的詳細配置與數據統計

## 數據增強方法詳解 🔧

本框架集成了多種數據增強方法，所有方法都同時對影像和其對應的遮罩進行相同的變換，以確保標註的有效性。

| 方法 (Method) | 描述 | 來源檔案 |
|---------------|------|----------|
| `brightness_contrast` | 隨機調整影像的亮度和對比度，模擬不同的光照條件 | `brightness_contrast.py` |
| `msr_clahe` | 結合多尺度視網膜增強 (MSR) 和對比度受限自適應直方圖均衡化 (CLAHE)，能有效增強影像的局部對比度和紋理細節，特別適合處理低對比度的衛星影像 | `msr_clahe.py` |
| `gaussian_blur` | 應用高斯模糊，用以模擬大氣或感測器造成的影像品質下降 | `gaussian_blur.py` |
| `polygon_erase` | 隨機在影像上擦除一個多邊形區域並填充隨機噪聲，模擬雲層遮蔽或其他物體遮擋 | `random_erasing.py` |
| `rotation_scaling` | 對影像進行隨機的旋轉和縮放，分為一般版本 (`rotation_scaling`) 和較溫和的版本 (`gentle_rotation`) | `rotation_scaling.py` |
| `crop_resize` | 隨機裁切影像的一部分並將其放大回原始尺寸，模擬不同的拍攝視角和距離。同樣提供一般版本 (`crop_resize`) 和溫和版本 (`gentle_crop`) | `crop_resize.py` |
| `cutmix` | 一種「安全」的CutMix實現。它會嘗試將一張影像的某個區塊貼到另一張影像上，但在貼上之前會檢查目標區域是否已存在重要的內波特徵，以避免破壞關鍵資訊 | `cutmix.py` |
| `wave_aware` | 一種感知內容的增強方法。它會先辨識出影像中的內波區域，然後對內波區域和背景區域採用不同的增強策略，例如只增強內波區域的對比度，而在背景區域進行更激進的變化 | `wave_aware.py` |

> **注意**：所有增強方法都在 `augmentation/__init__.py` 中被統一註冊到 `AUGMENTATION_METHODS` 字典，方便主程式透過字串名稱動態調用。

## 安装依赖 📦

建議在虛擬環境中安裝依賴：

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安裝依賴
pip install opencv-python
pip install numpy
pip install pillow
pip install scikit-image
pip install matplotlib
```

## 配置說明 ⚙️

### 實驗配置參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `name` | str | 實驗名稱，用於顯示和報告生成 |
| `description` | str | 實驗描述，說明實驗目的和方法 |
| `augmentation_factor` | float | 訓練集擴增倍數 |
| `val_augmentation_factor` | float | 驗證集擴增倍數 |
| `method_ratios` | dict | 各種增強方法的使用比例 |
| `method_params` | dict | 特定方法的客製化參數 |

### 增強方法參數

每種增強方法都支援客製化參數，詳細參數說明請參考各方法的源碼註解。

## 實驗報告 📊

每次實驗完成後會自動生成 `README.md` 報告，包含：

- 實驗基本信息（名稱、描述、時間）
- 數據集統計信息（原始和擴增後的數量）
- 使用的增強方法及其參數
- 數據分佈分析
- 實驗重現所需的完整配置

## 技術特點 🔬

- **模組化設計**：每種增強方法都是獨立的模組，易於擴展和維護
- **智能處理**：自動識別含有內波的影像，提高處理效率
- **參數化配置**：通過配置文件靈活控制實驗參數
- **自動化文檔**：實驗結果自動文檔化，確保可重現性
- **批次處理**：支持大規模數據集的批次增強處理

## 擴展指南 🔧

### 添加新的增強方法

1. 在 `augmentation/` 目錄下創建新的 Python 檔案
2. 實現增強函數，確保同時處理影像和遮罩
3. 在 `augmentation/__init__.py` 中註冊新方法
4. 在 `config.py` 中添加相應的配置

### 自定義評估指標

您可以修改 `readme_generator.py` 來添加更多的統計信息和分析圖表。

## 貢獻指南 🤝

歡迎對本專案提出改進建議或貢獻代碼！

### 如何貢獻

1. Fork 此專案
2. 建立您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

## 常見問題 ❓

### Q: 如何處理不同尺寸的影像？
A: 框架會自動處理不同尺寸的影像，但建議預先將數據集標準化到統一尺寸以獲得最佳效果。

### Q: 可以同時運行多個實驗嗎？
A: 可以，在 `config.py` 中定義多個實驗配置，程式會依序執行所有實驗。

### Q: 如何調整增強的強度？
A: 在 `method_params` 中為每個方法指定參數，或者調整 `method_ratios` 中的比例。

## 授權條款 📄

本專案採用 Apache License 2.0 授權。詳情請見 [LICENSE](LICENSE) 檔案。

**注意**：本框架專為海洋內波影像分割任務設計，使用前請確保您的數據集格式符合專案要求。所有增強方法都經過精心設計，以保持內波特徵的完整性。
