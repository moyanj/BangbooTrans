# BangbooTrans

![LSTM](https://img.shields.io/badge/Model-LSTM-blue) ![GPU Support](https://img.shields.io/badge/Support-GPU-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

BangbooTrans 是一个基于长短期记忆网络（LSTM）的邦布语翻译器。该项目旨在使用深度学习技术，实现从中文到邦布语的高质量翻译。

- ![Translation](https://img.shields.io/badge/Feature-Translation-blue) 使用LSTM模型进行翻译。
- ![GPU](https://img.shields.io/badge/Feature-GPU%20Support-green) 支持GPU训练，以加快模型训练速度。
- ![Logs](https://img.shields.io/badge/Feature-Logs-orange) 提供详细的训练日志和评估结果。
- ![Extendable](https://img.shields.io/badge/Feature-Extendable-brightgreen) 易于扩展和改进的代码架构。

## 项目描述

- 本项目是基于 `LSTM` 的 `seq2seq` 翻译模型。
- 代码中给出了许多详细的中文注释，方便大家更好地理解代码。
- 本项目可以实现中文到 绝区零 邦布语的翻译。

## 警告⚠️
由于邦布语数据集极小，若需加载大型数据集，请确保机器内存足够。（将一次性全部读取至内存）

## TODO

## 依赖安装

在开始使用之前，请确保已安装以下依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

请将您的数据集保存至`dataset`目录，且为`.csv`格式，第一列为邦布语，第二列为中文。

## 使用方法

### 配置参数

请修改 `config.py` 以配置模型训练和评估的参数。

### 训练模型

首先，确保你已经准备好了数据集并且配置好了参数。然后运行以下命令来训练模型：

```bash
python train.py
```

训练后，模型将保存于modelsmll
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


## 项目贡献

欢迎贡献代码和提出建议！你可以通过以下方式参与项目：

1. ![Fork](https://img.shields.io/badge/Step-1-blue) Fork 本仓库。
2. ![Branch](https://img.shields.io/badge/Step-2-blue) 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)。
3. ![Commit](https://img.shields.io/badge/Step-3-blue) 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)。
4. ![Push](https://img.shields.io/badge/Step-4-blue) 推送到分支 (`git push origin feature/AmazingFeature`)。
5. ![Pull Request](https://img.shields.io/badge/Step-5-blue) 提交一个 Pull Request。

## 许可证

该项目使用 MIT 许可证。附带的邦布语数据集使用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 进行许可。

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)

### 许可证条款

- **署名**：您必须给出适当的署名，提供指向本许可协议的链接，并指明是否进行了修改。您可以以任何合理的方式这样做，但不得以任何方式暗示许可人认可您或您的使用。
- **非商业性使用**：您不得将本数据集用于商业目的。
- **相同方式共享**：如果您对本数据集进行了修改、转换或增添，您必须基于与原先许可相同的许可分发您的贡献。

