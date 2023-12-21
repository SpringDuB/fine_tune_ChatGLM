---
license: Apache License 2.0
text:
  conversational:
    size_scale:
      - 100-10k
    type:
      - task-qa
  text-generation:
    language:
      - zh
  question-answering: {}

---
## 数据集描述
CoT数据集是通过对FLAN发布的9个CoT数据集进行格式化组合得到的。它包含9个CoT任务，涉及74771个样本。CoT_zh数据集是通过使用谷歌翻译将CoT数据集转成中文得到的。

### 数据集加载方式
```Python
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

# Load the CoT chinese dataset
ds_train = MsDataset.load('YorickHe/CoT_zh', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
print(next(iter(ds_train)))
```

### Clone with HTTP
```bash
git clone https://www.modelscope.cn/datasets/YorickHe/CoT_zh.git
```