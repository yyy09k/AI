# 多模态情感分析模型

## 项目简介

本项目实现了一个多模态情感分析模型，结合了 **图像** 和 **文本** 数据来进行情感分类。该模型采用了 **ResNet50** 提取图像特征，**RoBERTa** 提取文本特征，并通过融合图像和文本特征进行情感分类。实验代码在不同模型设置下进行了测试，包括仅图像模型、仅文本模型和多模态模型。

## 环境要求

该项目基于 **Python 3** 实现。运行代码前需要安装以下依赖项：

```bash
torch==1.10.0
torchvision==0.11.1
transformers==4.11.3
scikit-learn==0.24.2
pandas==1.3.3
numpy==1.21.2
tqdm==4.62.3
tensorboard==2.7.0
```

你可以通过以下命令安装所有依赖：

```bash
pip install -r requirements.txt
```

## 代码结构

项目文件结构如下：

```
|-- data/                        # 数据集文件夹
|-- train.txt                # 训练数据标签文件
|-- test_without_label.txt   # 测试数据标签缺失的文件
|
|-- runs/                        # TensorBoard 训练日志
|   |-- experiment_1/            # 多模态融合模型训练日志
|   |-- image_only_model/            # 仅图像模型训练日志
|   |-- text_only_model/            # 仅文本模型训练日志
|
|-- image_only_model.ipynb       # 仅图像模型训练代码
|-- multimodal_sentiment_analysis.ipynb # 多模态情感分析模型训练及测试代码
|-- text_only_model.ipynb        # 仅文本模型训练代码
|
|
|-- result.txt                   # 模型预测结果
|-- requirements.txt             # 环境依赖文件
|
|-- README.md                  #仓库解释文件
|-- 10225501437寇璟奕_AI_Lab5实验报告.pdf   #实验报告
```

## 执行流程

1. **准备数据**：请将您的数据集放置在 `data/` 文件夹中，并确保 `train.txt` 和 `test_without_label.txt` 文件格式正确。

2. **训练模型**：根据您的需求选择相应的模型进行训练。

   - **多模态模型**（图像 + 文本）：
     ```bash
     python multimodal_sentiment_analysis.ipynb
     ```
   
   - **仅图像模型**：
     ```bash
     python image_only_model.ipynb
     ```

   - **仅文本模型**：
     ```bash
     python text_only_model.ipynb
     ```

3. **查看训练结果**：运行后，TensorBoard 日志保存在 `runs/experiment_1` 目录下。你可以通过以下命令查看训练过程：

   ```bash
   tensorboard --logdir=runs
   ```

   打开浏览器访问 [http://localhost:6006](http://localhost:6006) 来查看训练过程的损失和准确率曲线。

## 参考库

本项目参考了以下开源库：

- **PyTorch**：用于构建和训练深度学习模型。
- **Transformers**：用于加载和使用预训练的 **RoBERTa** 模型。
- **Torchvision**：用于加载和处理图像数据。
- **Scikit-learn**：用于数据处理和评估模型性能。
- **Pandas**：用于数据加载和处理。
- **NumPy**：用于数值计算。
- **TQDM**：用于显示训练进度条。
- **TensorBoard**：用于可视化训练过程和性能指标。

## 运行示例

以下是运行多模态情感分析模型的示例命令：

```bash
python multimodal_sentiment_analysis.ipynb --train --epochs 5 --batch_size 32
```
## 贡献

欢迎提出问题、建议或贡献代码。如果您有任何问题，请提交 Issue 或直接创建 Pull Request。

## 引用

如果您在您的工作中使用了本代码，请引用以下参考文献：

1. [文章：深度学习模型和方法综述](https://zhuanlan.zhihu.com/p/661656067)
2. [文章：情感分析的深度学习方法](https://zhuanlan.zhihu.com/p/664335535)
3. [文章：基于深度学习的情感分析应用研究](https://journal.cuc.edu.cn/mediaCCUploadFiles/202204270314108f6ec5f5d91f4b88b19f26b11543ebc8.pdf)
4. [博客：深度学习模型在情感分析中的应用](https://blog.csdn.net/weixin_63595187/article/details/131679538)
