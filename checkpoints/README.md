# 模型檢查點

## 下載預訓練模型

### 從 Google Drive 下載

**[下載模型檔案](https://drive.google.com/file/d/1XZykFMfugoMCJKgaisC_4wnW5lV82-ZF/view?usp=sharing)**

### 模型資訊
- **檔案名稱**：`cifar100_final_model.pth`
- **檔案大小**：42.24 MB
- **測試準確率**：78.25% (Top-1), 94.85% (Top-5)
- **訓練 Epochs**：150
- **訓練時間**：150.2 分鐘 (Tesla T4)

### 使用方式

1. 從上方連結下載模型檔案
2. 將檔案放置於 `checkpoints/` 目錄
3. 使用以下程式碼載入模型：

```python
import torch
from models.resnet import CIFAR100CNN
from utils.config import TrainingConfig

# 載入檢查點
checkpoint = torch.load('checkpoints/cifar100_final_model.pth')

# 初始化模型
config = TrainingConfig()
model = CIFAR100CNN(config)

# 載入模型權重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 查看測試結果
print(f"測試準確率: {checkpoint['test_results']['test_acc']:.2f}%")
print(f"Top-5 準確率: {checkpoint['test_results']['test_acc_top5']:.2f}%")
```

### 檢查點內容

模型檔案包含以下資訊：
- `model_state_dict`: 模型權重
- `config`: 訓練配置
- `test_results`: 測試結果
- `model_stats`: 模型統計資訊
