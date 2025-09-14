# CIFAR-100 CNN 訓練系統

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuchen-0321/cifar100-cnn-training/blob/main/CIFAR_100_CNN_Training.ipynb)

## 計算智慧期末專案

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-78.38%25-brightgreen.svg)

**學生**：李泓斌
**學號**：C111110141
**任課教授**：曾建誠
**課程**：計算智慧

---

## 專案成果

一個專為 CIFAR-100 分類任務設計的CNN 訓練系統，達到 **78.23% 驗證準確率**和 **94.86% Top-5 準確率**。

## 效能亮點

- **測試準確率**: 78.25% (Top-1), 94.85% (Top-5)
- **訓練時間**: 150.2 分鐘（Tesla T4 GPU）
- **模型大小**: 42.24 MB（11M 參數）
- **推論速度**: 2300+ 張圖片/秒（Tesla T4）
- **GPU 記憶體峰值**: 3.54 GB

## 訓練結果

### 最終指標
| 指標 | 數值 |
|------|------|
| **最佳驗證準確率** | 78.23%（第 150 epoch） |
| **最終測試準確率（Top-1）** | 78.25% |
| **最終測試準確率（Top-5）** | 94.85% |
| **平均類別準確率** | 78.25% |
| **類別準確率標準差** | 10.56% |
| **最低類別準確率** | 53.00% |
| **最高類別準確率** | 94.00% |

## 快速開始

### Google Colab 環境

```python
# 複製儲存庫
!git clone https://github.com/yourusername/cifar100-cnn-training.git
%cd cifar100-cnn-training

# 使用預設設定執行訓練
!python train.py
```

### 本地環境

```bash
# 複製儲存庫
git clone https://github.com/yourusername/cifar100-cnn-training.git
cd cifar100-cnn-training

# 安裝依賴套件
pip install -r requirements.txt

# 執行訓練
python train.py --epochs 150 --batch-size 256 --lr 0.1
```

## 核心功能

### 系統架構
- **模組化設計**：清晰的關注點分離，包含資料載入、模型架構、訓練邏輯和監控模組
- **生產就緒**：完整的錯誤處理、資源管理和自動恢復機制
- **記憶體效率**：混合精度訓練、梯度累積和動態批次大小調整

### 訓練最佳化
- **混合精度訓練**：2-3 倍速度提升，精度損失最小
- **學習率調度**：餘弦退火配合暖身，確保穩定收斂
- **進階資料增強**：三級增強策略（輕度/中度/強度）
- **標籤平滑**：防止過度自信預測，改善泛化能力
- **早停機制**：自動終止訓練以防止過擬合

### 模型架構
- **ResNet 啟發設計**：高效的殘差塊配合批次正規化
- **漸進式通道擴展**：64 → 128 → 256 → 512 通道
- **全域平均池化**：減少參數同時維持效能
- **Dropout 正規化**：可配置的 dropout 防止過擬合

## 專案結構

```
cifar100-cnn-training/
├── train.py                 # 主訓練腳本
├── models/
│   ├── __init__.py
│   └── resnet.py           # 模型架構
├── data/
│   ├── __init__.py
│   └── dataloader.py       # 資料管線和增強
├── trainers/
│   ├── __init__.py
│   └── trainer.py          # 訓練協調器
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 指標追蹤和視覺化
│   ├── monitor.py          # 效能監控
│   └── config.py           # 配置管理
├── configs/
│   ├── default.yaml        # 預設配置
│   └── experiments/        # 實驗配置
├── checkpoints/            # 模型檢查點（自動建立）
├── logs/                   # 訓練日誌（自動建立）
├── requirements.txt        # Python 依賴
└── README.md              # 本文件
```

## 訓練配置

### 預設設定

```python
@dataclass
class TrainingConfig:
    # 模型參數
    num_classes: int = 100
    dropout_rate: float = 0.3
    
    # 訓練參數
    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # 最佳化
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    
    # 調度器
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # 資料增強
    augmentation_strength: str = 'moderate'
```

### 自訂配置

```python
from train import train_cifar100

# 自訂訓練
model, metrics, results = train_cifar100(
    epochs=150,
    batch_size=256,
    learning_rate=0.1,
    mixed_precision=True,
    augmentation='moderate'
)
```

## 訓練曲線

訓練過程呈現三個明顯階段：

1. **快速初始學習**（Epoch 1-30）：從隨機猜測快速提升
2. **穩定改進**（Epoch 30-100）：持續但較慢的進步
3. **精細調整**（Epoch 100-150）：微調至最佳效能

### 關鍵里程碑
- Epoch 10: 39.71% 驗證準確率
- Epoch 30: 59.55% 驗證準確率
- Epoch 50: 64.59% 驗證準確率
- Epoch 100: 78.40% 驗證準確率
- Epoch 150: 78.23% 驗證準確率（最佳）

## 系統需求

### 硬體
- **GPU**：具有 4GB+ VRAM 的 NVIDIA GPU（在 Tesla T4 上測試）
- **RAM**：最少 8GB
- **儲存空間**：2GB 用於資料集和檢查點

### 軟體
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（用於 GPU 加速）

## 依賴套件

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.65.0
```

## 使用範例

### 基本訓練

```python
from cifar100_pipeline import CIFAR100Pipeline

# 初始化管線
pipeline = CIFAR100Pipeline()

# 執行訓練
model, metrics, results = pipeline.run()
```

### 載入預訓練模型

```python
import torch
from models.resnet import CIFAR100CNN
from utils.config import TrainingConfig

# 載入檢查點
checkpoint = torch.load('checkpoints/cifar100_final_model.pth')

# 初始化模型
config = TrainingConfig()
model = CIFAR100CNN(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 模型已準備好進行推論
print(f"測試準確率: {checkpoint['test_results']['test_acc']:.2f}%")
```

### 推論範例

```python
import torch
from torchvision import transforms
from PIL import Image

# 準備圖片
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761))
])

# 載入並預處理圖片
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# 推論
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
```

## 實驗追蹤

### 訓練日誌

系統自動追蹤：
- 損失曲線（訓練和驗證）
- 準確率指標（Top-1 和 Top-5）
- 學習率調度
- 每類別準確率統計
- GPU 記憶體使用
- 推論吞吐量

### 檢查點

每 5 個 epoch 自動儲存檢查點，包含：
- 模型權重
- 優化器狀態
- 學習率調度器狀態
- 訓練配置
- 效能指標

## 效能基準

### 推論效能（Tesla T4）

| 批次大小 | 吞吐量（img/s） | 記憶體（GB） |
|---------|----------------|-------------|
| 32 | 2,162.7 | 0.05 |
| 64 | 2,260.9 | 0.05 |
| 128 | 2,308.2 | 0.05 |
| 256 | 2,305.8 | 0.05 |

### 與其他方法比較

| 方法 | Top-1 準確率 | Top-5 準確率 | 參數量 |
|-----|-------------|-------------|--------|
| **我們的方法** | **78.25%** | **94.85%** | 11M |
| ResNet-18 | 75.61% | 93.45% | 11M |
| VGG-16 | 71.59% | 90.49% | 138M |
| DenseNet-121 | 79.04% | 94.84% | 8M |

## 技術洞察

### 關鍵設計決策

1. **殘差連接**：解決梯度消失，實現更深的網路
2. **全域平均池化**：相比全連接層減少 90% 參數
3. **混合精度訓練**：2-3 倍速度提升，準確率損失 <0.1%
4. **餘弦退火**：平滑的學習率衰減防止突然的效能下降
5. **標籤平滑（0.1）**：改善泛化能力 1-2% 準確率

### 最佳化策略

- **梯度裁剪**：防止梯度爆炸
- **批次正規化**：加速收斂
- **Dropout（0.3）**：正規化而不過度減少容量
- **資料增強**：平衡的增強防止過擬合

## 訓練技巧

### 獲得最佳結果

1. **使用 150 epochs**：100 epochs 後仍有改進空間
2. **批次大小 256**：在速度和泛化之間取得平衡
3. **中度增強**：避免過度增強造成的噪音
4. **監控驗證損失**：早停防止過擬合

### 常見問題解決

**過擬合問題**：
- 增加 dropout rate（0.3 → 0.5）
- 使用強度資料增強
- 啟用更多正規化技術

**收斂緩慢**：
- 檢查學習率是否過小
- 確認批次正規化正常運作
- 考慮使用更大的批次大小

**GPU 記憶體不足**：
- 降低批次大小
- 啟用梯度累積
- 使用混合精度訓練

## 授權

MIT License - 詳見 [LICENSE](LICENSE) 檔案

## 聯絡方式

**國立高雄科技大學電腦與通訊工程系**   
**學生**：李泓斌  
**學號**：C111110141  
**Email**：[c111110141@nkust.edu.tw](mailto:c111110141@nkust.edu.tw)  
如有問題或建議，請在 GitHub 上開啟 issue。
