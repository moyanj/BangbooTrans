# BangbooTrans

BangbooTrans 是一个基于长短期记忆网络（LSTM）的邦布语翻译器。该项目旨在使用深度学习技术，实现从源语言到邦布语的高质量翻译。

## 功能特性

- 使用LSTM模型进行翻译。
- 支持GPU训练，以加快模型训练速度。
- 提供详细的训练日志和评估结果。
- 易于扩展和改进的代码架构。

## 依赖安装

在开始使用之前，请确保已安装以下依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

请直接写入`邦布语数据集.csv`,第一列为邦布语，第二列为中文
## 使用方法

### 配置参数
请修改`config.py`

### 训练模型

首先，确保你已经准备好了数据集并且配置好了参数。然后运行以下命令来训练模型：

```bash
python train.py
```

### 评估模型

训练完成后，你可以使用以下命令评估模型性能：

```bash
python test.py
```

### 翻译文本

你可以使用训练好的模型进行翻译：

```python
from inference import Inference

translator = Inference('<模型名称>')
source_text = "你的源语言文本"
translation = translator.eval(source_text)
print(translation)
```

## 文件结构

```plaintext
├── 邦布语数据集.csv
├── config.py
├── dataset.py
├── Deploy
│   └── onnx.jsTest.html
├── inference
│   ├── dataset.py
│   ├── __init__.py
│   ├── model.py
├── model.py
├── requirements.txt
├── test.py
├── train.py
├── readme.md
└── webui.py
```

## 项目贡献

欢迎贡献代码和提出建议！你可以通过以下方式参与项目：

1. Fork 本仓库。
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)。
4. 推送到分支 (`git push origin feature/AmazingFeature`)。
5. 提交一个 Pull Request。

## 许可证

该项目使用 MIT 许可证。附带的邦布语数据集使用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 进行许可。
