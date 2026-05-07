# 基于Transformer的多雷达干扰样式识别
使用Transformer模型对多个雷达接收的干扰样式进行识别，重点在于数据融合处理

# 快速跑通（本地最小复现）

## 1) 安装依赖
```bash
python -m pip install timm h5py
```

## 2) 生成模拟数据（用于验证训练链路）
```bash
python generate_synthetic_mat_data.py --out_dir synthetic_data --num_classes 3 --samples_per_class 12 --seq_len 20 --num_nodes 3
```

### 更容易学到模式的结构化干扰数据（推荐）
```bash
python generate_synthetic_mat_data.py \
  --out_dir synthetic_data_structured12_long \
  --mode structured \
  --save_format complex \
  --num_classes 12 \
  --samples_per_class 80 \
  --seq_len 64 \
  --num_periods 8 \
  --num_nodes 3 \
  --seed 2026
```
说明：
- `--mode structured`：按干扰模式生成类别（不是纯随机噪声）。
- `--save_format complex`：生成速度更快，且现有读取器已支持 complex。

## 3) 跑通 v2 训练入口
```bash
python train_main-TR-v2.py \
  --datapath synthetic_data \
  --arch scheme1 \
  --epochs 1 \
  --iterTime 1 \
  --batch_size 4 \
  --train_size 0.6 \
  --num_classes 3 \
  --seq_len 20 \
  --patch_size 4 \
  --stride 4 \
  --d_model 32 \
  --mlp_ratio 2 \
  --nhead 4 \
  --num_blocks 1 \
  --noCuda
```

### 结构化 12 类数据的推荐训练命令（效果明显优于随机数据）
```bash
python train_main-TR-v2.py \
  --datapath synthetic_data_structured12_long \
  --arch scheme1 \
  --epochs 15 \
  --iterTime 1 \
  --batch_size 32 \
  --train_size 0.7 \
  --num_classes 12 \
  --seq_len 64 \
  --patch_size 8 \
  --stride 8 \
  --d_model 96 \
  --mlp_ratio 2 \
  --nhead 4 \
  --num_blocks 2 \
  --noCuda
```

## 4) 跑通 v3 训练入口（Scheme6）
```bash
python train_main-TR-v3.py \
  --datapath synthetic_data \
  --epochs 1 \
  --iterTime 1 \
  --batch_size 4 \
  --train_size 0.6 \
  --val_size 0.1 \
  --num_classes 3 \
  --seq_len 20 \
  --patch_size 4 \
  --stride 4 \
  --d_model 32 \
  --mlp_ratio 2 \
  --nhead 4 \
  --num_blocks 1 \
  --noCuda
```

# 数据处理

不失一般性，数据的实虚部当作2个通道处理，分块不重合（也可以分块重合进行试验）

# 论文风格图复现（数据图 + 结果图）

可用下面脚本直接生成常见论文图：IQ波形、星座图、时频图、训练曲线、混淆矩阵、各类准确率柱状图。

## 1) 从 `.mat` 数据生成图
```bash
python tools/plot_paper_figures.py \
  --data_dir synthetic_data \
  --class_file class_00.mat \
  --sample_idx 0 \
  --node_idx 0 \
  --out_dir figures_data
```

## 2) 从训练日志生成图
```bash
python tools/plot_paper_figures.py \
  --log_path log-TR-v3/scheme6-iterTime-1-heads-4-lr-0.001-epochs-1-Time-20260308-024858-log.txt \
  --out_dir figures_log
```

## 3) 数据图 + 结果图一起生成
```bash
python tools/plot_paper_figures.py \
  --data_dir synthetic_data \
  --log_path log-TR-v2/scheme1-iterTime-1-heads-4-lr-0.001-epochs-1-Time-20260308-024610-log.txt \
  --out_dir figures_all
```

> 若你的真实数据中时间/样本/节点轴顺序不同，可用参数 `--time_axis --sample_axis --node_axis` 调整。

# 包含模型 （以三个雷达接收干扰数据为例）：

-使用单个节点的数据各自独立训练Transformer，各自预测结果

-3个节点的数据先通过可学习的权重进行线性相加，然后送入一个共享Transformer

-3个节点的数据先通过节点平均进行线性相加，然后送入一个共享Transformer

-3个节点的数据送入共享一个Transformer，最后提取的特征加权融合。

-3节点各用独立Transformer→特征加权融合。

-维度分离注意力（节点维+分组维）

-维度分离注意力 + 全局融合向量
