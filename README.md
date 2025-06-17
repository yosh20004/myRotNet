# RotNet 项目说明

## 项目结构

```
RotNet/
├── check_duplicates.py         # 检查数据重复的脚本
├── model/                      # （可选）模型权重文件
├── predict/                    # 预测相关脚本
├── rename_images.py            # 重命名图片脚本
├── requirements.txt            # 依赖包列表
├── results/                    # （可选）结果图片
├── results_enhanced/           # （可选）增强结果图片
├── train/                      # 训练相关代码
└── my_cat.jpg                  # （可选）示例图片
```

## 环境准备

建议使用 Python 3.8 及以上版本。

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. （可选）建议使用虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
```

## 数据准备

请将您的数据集（如 CIFAR-10）放在 `data/` 目录下，结构如下：

```
data/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── ...
```

如需自定义数据，请参考 `train/` 目录下的代码，调整数据加载部分。

## 训练模型

进入 `train/` 目录，运行训练脚本，例如：

```bash
python train/finetune_advanced_v2.py
```

## 预测与评估

预测脚本位于 `predict/` 目录，例如：

```bash
python predict/predict_on_my_testset.py
```

## 其他说明

- `model/` 目录下为训练好的模型权重（如有需要可上传）
- `results/`、`results_enhanced/` 为结果图片（可选上传）
- `data/` 目录不建议上传，数据请在目标机器自行准备

如有问题请联系项目作者。 