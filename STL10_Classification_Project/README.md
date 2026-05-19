# STL-10 Image Classification Project

基于 STL-10 风格目录数据（`train/` + `test/` + 每类子文件夹）实现的图像分类工程，包含：

- 基础 CNN（3 个卷积块 + 池化 + 两层全连接）
- 可扩展模型注册表（支持 `BaseCNN` 与 `ResNetLite`）
- AMP 训练流程
- Train/Valid 双曲线绘制
- Test 集分类报告与混淆矩阵

## 1. 目录要求

请将数据放到以下目录（默认配置）：

```text
STL10_Classification_Project/
└── data/
    ├── train/
    │   ├── airplane/
    │   ├── bird/
    │   └── ...
    └── test/
        ├── airplane/
        ├── bird/
        └── ...
```

其中 `train` 必须总计 7000 张图像，程序会严格按 `5600/1400` 划分 train/valid；`test` 为 1000 张图像。

## 2. 安装依赖

```bash
pip install torch torchvision pyyaml matplotlib scikit-learn seaborn
```

## 3. 训练

```bash
python train.py --config configs/base_cnn.yaml
```

训练输出在 `results/`（或对应实验配置中的 `output_dir`）：

- `best_base_model.pth`（或配置名）
- `loss_curve.png`
- `accuracy_curve.png`
- `history.json`

## 4. 评估

```bash
python evaluate.py --config configs/base_cnn.yaml
```

评估输出：

- `classification_report.txt`（Precision / Recall / F1-score / macro avg）
- `confusion_matrix.png`

## 5. 可扩展实验

- 数据增强：`configs/exp_a_aug.yaml`
- 架构替换（ResNetLite）：`configs/exp_b_resnet.yaml`
- 优化器替换（AdamW）：`configs/exp_c_adamw.yaml`
