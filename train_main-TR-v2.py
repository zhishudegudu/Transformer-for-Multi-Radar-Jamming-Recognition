import argparse
import sys
import time
import os
from sklearn.metrics import confusion_matrix
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

# 导入正确的数据读取和模型模块
from data_read import load_data, get_dataloaders
from data_read_v1 import load_data, get_dataloaders, get_data
from model import RadarInterferenceModel_Scheme1, RadarInterferenceModel_Scheme2, RadarInterferenceModel_Scheme3, RadarInterferenceModel_Scheme4, RadarInterferenceModel_Scheme5

# 增加程序运行效率
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True  # 注意如果是CNN网络，这个要设置为False

parser = argparse.ArgumentParser(description='PyTorch Radar Interference Recognition')
# '路径设置'
parser.add_argument('--datapath', type=str, default=r'F:/Radar_data/DATA/12种干扰样式数据/', help='training dataset file path')
parser.add_argument('--logDir', default='./log-TR-v2', type=str, metavar='PATH', help='path to latest checkpoint')

# '训练参数设置'
parser.add_argument('--saveMode', default='valAcc', type=str, help='train model setting')
parser.add_argument('--searchParm', type=int, default=0, metavar='searchParm', help='1 means search parameters')
parser.add_argument('--iterTime', type=int, default=5, metavar='iterTime', help='the iteration time')
parser.add_argument('--arch', default='scheme1', choices=['scheme1', 'scheme2', 'scheme3', 'scheme4', 'scheme5'], type=str, help='architecture to use')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--startEpoch', default=0, type=int, metavar='N', help='manual epoch number')
parser.add_argument('--early_stop_patience', type=int, default=5, help='验证集无提升时的早停轮数，0表示关闭')

# '其他设置'
parser.add_argument('--noCuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--cuda', type=bool, default=False, help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--train_size', type=float, default=100, help='训练集比例或数量')

# '模型参数'
parser.add_argument('--input_dim', type=int, default=2, help='输入维度（实部+虚部）')
parser.add_argument('--input_period', type=int, default=8, help='雷达周期数')
parser.add_argument('--patch_size', type=int, default=16, help='Token分块尺寸')
parser.add_argument('--stride', type=int, default=16, help='卷积步长')
parser.add_argument('--seq_len', type=int, default=2000, help='单脉冲周期采样点')
parser.add_argument('--num_nodes', type=int, default=3, help='雷达视图数')
parser.add_argument('--d_model', type=int, default=128, help='特征维度')
parser.add_argument('--mlp_ratio', type=int, default=2, help='隐藏层维度倍率')
parser.add_argument('--nhead', type=int, default=4, help='注意力头数')
parser.add_argument('--num_blocks', type=int, default=4, help='编码模块堆叠数')
parser.add_argument('--num_classes', type=int, default=12, help='干扰样式类别数')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
parser.add_argument('--cls_head', type=str, default='linear', choices=['linear', 'cosine'], help='分类头类型')
parser.add_argument('--cosine_scale', type=float, default=10.0, help='cosine classifier缩放系数')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='交叉熵标签平滑系数')
parser.add_argument('--fusion_type', type=str, default='weighted', choices=['weighted', 'average'], help='节点融合类型')
parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal', 'amsoftmax', 'arcface'], help='损失函数类型')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
parser.add_argument('--am_margin', type=float, default=0.35, help='AM-Softmax additive margin')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值，0表示关闭')
parser.add_argument('--aug_scale', type=float, default=0.15, help='IQ幅度缩放范围')
parser.add_argument('--aug_phase_max', type=float, default=0.35, help='IQ相位旋转最大弧度')
parser.add_argument('--aug_time_shift', type=int, default=4, help='训练时序平移最大步长')
parser.add_argument('--aug_noise_std', type=float, default=0.03, help='训练噪声增强强度')
parser.add_argument('--aug_snr_min', type=float, default=None, help='按SNR采样的AWGN增强下限(dB)，None表示关闭')
parser.add_argument('--aug_snr_max', type=float, default=None, help='按SNR采样的AWGN增强上限(dB)，None表示关闭')
parser.add_argument('--aug_snr_prob', type=float, default=0.0, help='按SNR采样AWGN增强概率')
parser.add_argument('--node_dropout_prob', type=float, default=0.15, help='随机屏蔽单个节点的概率')
parser.add_argument('--center_loss_weight', type=float, default=0.0, help='类中心约束权重')
parser.add_argument('--center_lr', type=float, default=0.5, help='类中心更新学习率')
parser.add_argument('--class_weight_boost_cls', type=int, default=-1, help='需要额外加权的类别索引，-1表示关闭')
parser.add_argument('--class_weight_boost', type=float, default=1.0, help='指定类别的交叉熵权重放大倍数')
parser.add_argument('--resume', type=str, default='', help='从已有checkpoint恢复模型参数')
parser.add_argument('--use_denoise_stem', action='store_true', default=False, help='在scheme5前增加IQ去噪stem')
parser.add_argument('--use_tf_cross_attn', action='store_true', default=False, help='在scheme5中启用时域到频域的单向token级交叉注意力')
parser.add_argument('--domain_fusion_type', type=str, default='gate', choices=['gate', 'gated_interaction', 'cross_gated_interaction'], help='节点内时频融合方式')
parser.add_argument('--use_node_co_attn', action='store_true', default=False, help='在scheme5中启用节点维协同注意力')
parser.add_argument('--use_node_reliability', action='store_true', default=False, help='在scheme5中启用显式节点可靠性估计')
parser.add_argument('--node_contrastive_weight', type=float, default=0.0, help='跨节点一致性/对比约束权重')
parser.add_argument('--node_contrastive_temp', type=float, default=0.2, help='跨节点对比损失温度')
parser.add_argument('--node_consistency_weight', type=float, default=0.0, help='可靠性加权的跨节点一致性约束权重')
parser.add_argument('--distill_teacher_ckpt', type=str, default='', help='蒸馏教师模型checkpoint路径')
parser.add_argument('--distill_clean_datapath', type=str, default='', help='蒸馏时clean数据路径（与噪声数据一一对应）')
parser.add_argument('--distill_alpha', type=float, default=0.5, help='蒸馏KL损失权重')
parser.add_argument('--distill_beta', type=float, default=0.05, help='蒸馏特征MSE损失权重')
parser.add_argument('--distill_temp', type=float, default=2.0, help='蒸馏温度')
parser.add_argument('--teacher_use_denoise_stem', action='store_true', default=False, help='教师模型是否启用去噪stem')

# '优化器参数'
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

args = parser.parse_args()
# args, unknown = parser.parse_known_args()
args.cuda = not args.noCuda and torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 设置日志路径
args.time = time.strftime("%Y%m%d-%H%M%S")
if args.searchParm:
    args.logTxtPath = os.path.join(args.logDir, f"{args.arch}-Time-{args.time}-log.txt")
else:
    args.logTxtPath = os.path.join(args.logDir, f"{args.arch}-iterTime-{args.iterTime}-heads-{args.nhead}-lr-{args.lr}-epochs-{args.epochs}-Time-{args.time}-log.txt")

# 创建日志目录
os.makedirs(args.logDir, exist_ok=True)

# 配置日志
logFormat = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=logFormat, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(args.logTxtPath)
fh.setFormatter(logging.Formatter(logFormat))
logging.getLogger().addHandler(fh)

# 记录参数信息
logging.info('batch_size %d', args.batch_size)
logging.info('epochs %d', args.epochs)
logging.info('iterTime %d', args.iterTime)
logging.info('lr %f', args.lr)
logging.info('d_model %d', args.d_model)
logging.info('patch_size %d', args.patch_size)
logging.info('num_blocks %d', args.num_blocks)
logging.info('nhead %d', args.nhead)
logging.info('mlp_ratio %d', args.mlp_ratio)
logging.info('dropout %f', args.dropout)
logging.info('cls_head %s', args.cls_head)
logging.info('cosine_scale %f', args.cosine_scale)
logging.info('label_smoothing %f', args.label_smoothing)
logging.info('early_stop_patience %d', args.early_stop_patience)
logging.info('architecture: %s', args.arch)
logging.info('fusion_type: %s', args.fusion_type)
logging.info('loss_type: %s', args.loss_type)
logging.info('focal_gamma %f', args.focal_gamma)
logging.info('am_margin %f', args.am_margin)
logging.info('weight_decay %f', args.weight_decay)
logging.info('grad_clip %f', args.grad_clip)
logging.info('aug_scale %f', args.aug_scale)
logging.info('aug_phase_max %f', args.aug_phase_max)
logging.info('aug_time_shift %d', args.aug_time_shift)
logging.info('aug_noise_std %f', args.aug_noise_std)
logging.info('aug_snr_min %s', str(args.aug_snr_min))
logging.info('aug_snr_max %s', str(args.aug_snr_max))
logging.info('aug_snr_prob %f', args.aug_snr_prob)
logging.info('node_dropout_prob %f', args.node_dropout_prob)
logging.info('center_loss_weight %f', args.center_loss_weight)
logging.info('center_lr %f', args.center_lr)
logging.info('class_weight_boost_cls %d', args.class_weight_boost_cls)
logging.info('class_weight_boost %f', args.class_weight_boost)
logging.info('resume %s', args.resume if args.resume else 'None')
logging.info('use_denoise_stem %s', str(args.use_denoise_stem))
logging.info('use_tf_cross_attn %s', str(args.use_tf_cross_attn))
logging.info('domain_fusion_type %s', args.domain_fusion_type)
logging.info('use_node_co_attn %s', str(args.use_node_co_attn))
logging.info('use_node_reliability %s', str(args.use_node_reliability))
logging.info('node_contrastive_weight %f', args.node_contrastive_weight)
logging.info('node_contrastive_temp %f', args.node_contrastive_temp)
logging.info('node_consistency_weight %f', args.node_consistency_weight)
logging.info('distill_teacher_ckpt %s', args.distill_teacher_ckpt if args.distill_teacher_ckpt else 'None')
logging.info('distill_clean_datapath %s', args.distill_clean_datapath if args.distill_clean_datapath else 'None')
logging.info('distill_alpha %f', args.distill_alpha)
logging.info('distill_beta %f', args.distill_beta)
logging.info('distill_temp %f', args.distill_temp)
logging.info('teacher_use_denoise_stem %s', str(args.teacher_use_denoise_stem))

# 全局变量
train_loss_list = []
val_loss_list = []


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class AMSoftmaxLoss(torch.nn.Module):
    def __init__(self, margin: float = 0.35, scale: float = 10.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, target):
        margin_logits = logits.clone()
        margin_logits[torch.arange(target.size(0), device=target.device), target] -= self.margin * self.scale
        return F.cross_entropy(margin_logits, target)


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, margin: float = 0.30, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, target):
        cosine = torch.clamp(logits / self.scale, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_cos = torch.cos(theta[torch.arange(target.size(0), device=target.device), target] + self.margin)
        margin_logits = logits.clone()
        margin_logits[torch.arange(target.size(0), device=target.device), target] = target_cos * self.scale
        return F.cross_entropy(margin_logits, target)


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


class PairedRadarDataset(Dataset):
    """返回(noisy, clean, label)的配对数据集，用于蒸馏训练。"""
    def __init__(self, noisy_data, clean_data, labels):
        if noisy_data.shape != clean_data.shape:
            raise ValueError(f"noisy/clean shape mismatch: {noisy_data.shape} vs {clean_data.shape}")
        if noisy_data.shape[0] != labels.shape[0]:
            raise ValueError("sample/label size mismatch in PairedRadarDataset")
        self.noisy_data = noisy_data
        self.clean_data = clean_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.noisy_data[idx], self.clean_data[idx], self.labels[idx]


def get_paired_dataloaders(noisy_all_data, clean_all_data, args):
    (Xn_train, y_train), (Xn_test, y_test), (Xn_val, y_val) = load_data(noisy_all_data, args)
    (Xc_train, y_train_c), (Xc_test, y_test_c), (Xc_val, y_val_c) = load_data(clean_all_data, args)

    if not torch.equal(y_train, y_train_c):
        raise ValueError("train labels are not aligned between noisy and clean data.")
    if not torch.equal(y_test, y_test_c):
        raise ValueError("test labels are not aligned between noisy and clean data.")
    if not torch.equal(y_val, y_val_c):
        raise ValueError("val labels are not aligned between noisy and clean data.")

    train_dataset = PairedRadarDataset(Xn_train, Xc_train, y_train)
    test_dataset = PairedRadarDataset(Xn_test, Xc_test, y_test)
    val_dataset = PairedRadarDataset(Xn_val, Xc_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    return train_loader, test_loader, val_loader


def build_criterion(args):
    class_weights = None
    if 0 <= args.class_weight_boost_cls < args.num_classes and args.class_weight_boost != 1.0:
        class_weights = torch.ones(args.num_classes, dtype=torch.float32)
        class_weights[args.class_weight_boost_cls] = float(args.class_weight_boost)
        logging.info('启用类别加权: class_%02d x %.3f', args.class_weight_boost_cls, args.class_weight_boost)
        if args.cuda:
            class_weights = class_weights.cuda()
    if args.loss_type == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.loss_type == 'amsoftmax':
        criterion = AMSoftmaxLoss(margin=args.am_margin, scale=args.cosine_scale)
    elif args.loss_type == 'arcface':
        criterion = ArcFaceLoss(margin=args.am_margin, scale=args.cosine_scale)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    return criterion.cuda() if args.cuda else criterion


def cross_node_contrastive_loss(node_features, temperature: float = 0.2):
    if node_features is None or node_features.ndim != 3 or node_features.shape[1] < 2:
        return None

    batch_size, num_nodes, feat_dim = node_features.shape
    feats = F.normalize(node_features.reshape(batch_size * num_nodes, feat_dim), dim=1)
    sim = torch.matmul(feats, feats.t()) / temperature
    sim = sim - torch.max(sim, dim=1, keepdim=True).values.detach()

    sample_ids = torch.arange(batch_size, device=node_features.device).repeat_interleave(num_nodes)
    pos_mask = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)
    eye_mask = torch.eye(batch_size * num_nodes, device=node_features.device, dtype=torch.bool)
    pos_mask = pos_mask & (~eye_mask)

    exp_sim = torch.exp(sim) * (~eye_mask)
    denom = exp_sim.sum(dim=1) + 1e-8
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    valid = pos_mask.sum(dim=1) > 0
    if not valid.any():
        return None
    loss = -torch.log((pos_sum[valid] + 1e-8) / denom[valid])
    return loss.mean()


def reliability_weighted_consistency_loss(node_features, node_weights=None):
    if node_features is None or node_features.ndim != 3 or node_features.shape[1] < 2:
        return None
    if node_weights is None:
        weights = torch.full(
            (node_features.shape[0], node_features.shape[1]),
            1.0 / node_features.shape[1],
            device=node_features.device,
            dtype=node_features.dtype,
        )
    else:
        weights = node_weights / node_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ref = torch.sum(node_features * weights.unsqueeze(-1), dim=1, keepdim=True)
    sq = ((node_features - ref) ** 2).mean(dim=-1)
    return torch.sum(weights * sq, dim=1).mean()


def augment_iq_batch(data):
    if data.ndim != 5:
        return data

    x = data.clone()
    batch_size, num_nodes, _, _, _ = x.shape
    device = x.device

    if args.aug_scale > 0:
        scale = 1.0 + (2.0 * torch.rand((batch_size, num_nodes, 1, 1, 1), device=device) - 1.0) * args.aug_scale
        x = x * scale

    if args.aug_phase_max > 0:
        theta = (2.0 * torch.rand((batch_size, num_nodes, 1, 1, 1), device=device) - 1.0) * args.aug_phase_max
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        i = x[:, :, :, 0:1, :]
        q = x[:, :, :, 1:2, :]
        x = torch.cat((cos_t * i - sin_t * q, sin_t * i + cos_t * q), dim=3)

    if args.aug_time_shift > 0:
        shifts = torch.randint(-args.aug_time_shift, args.aug_time_shift + 1, (batch_size, num_nodes), device=device)
        for b in range(batch_size):
            for n in range(num_nodes):
                shift = int(shifts[b, n].item())
                if shift != 0:
                    x[b, n] = torch.roll(x[b, n], shifts=shift, dims=0)

    if args.node_dropout_prob > 0:
        drop_samples = torch.rand(batch_size, device=device) < args.node_dropout_prob
        if drop_samples.any():
            drop_nodes = torch.randint(0, num_nodes, (batch_size,), device=device)
            for b in torch.nonzero(drop_samples, as_tuple=False).flatten():
                x[b, drop_nodes[b]] = 0.0

    if args.aug_noise_std > 0:
        x = x + torch.randn_like(x) * args.aug_noise_std

    # SNR-based complex AWGN augmentation for low-SNR robustness.
    if (
        args.aug_snr_min is not None
        and args.aug_snr_max is not None
        and args.aug_snr_prob > 0
        and torch.rand(1, device=device).item() < args.aug_snr_prob
    ):
        i = x[:, :, :, 0, :]
        q = x[:, :, :, 1, :]
        xc = torch.complex(i, q)
        sig_power = torch.mean(torch.abs(xc) ** 2, dim=(2, 3), keepdim=True) + 1e-8
        snr_db = torch.empty((batch_size, num_nodes, 1, 1), device=device).uniform_(args.aug_snr_min, args.aug_snr_max)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = (
            torch.randn_like(xc.real) + 1j * torch.randn_like(xc.real)
        ) * torch.sqrt(noise_power / 2.0)
        xc = xc + noise
        x = torch.stack((xc.real, xc.imag), dim=3)

    return x

def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], None, batch[1]
    raise ValueError("Unsupported batch format.")


def train(
    epoch,
    model,
    train_loader,
    optimizer,
    criterion,
    center_criterion=None,
    center_optimizer=None,
    teacher_model=None,
    distill_cfg=None,
):
    """训练函数"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        data, clean_data, target = unpack_batch(batch)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            if clean_data is not None:
                clean_data = clean_data.cuda()

        data = augment_iq_batch(data)

        # 获取模型输出
        node_features = None
        node_weights = None
        if (center_criterion is not None or args.node_contrastive_weight > 0 or args.node_consistency_weight > 0) and hasattr(model, 'forward_with_details'):
            outputs, features, node_features, node_weights = model.forward_with_details(data)
        elif center_criterion is not None and hasattr(model, 'forward_with_features'):
            outputs, features = model.forward_with_features(data)
        else:
            outputs = model(data)
            features = None
        
        # 计算分类损失
        loss = criterion(outputs, target)

        # 蒸馏损失（clean teacher -> noisy student）
        if teacher_model is not None and clean_data is not None:
            with torch.no_grad():
                if hasattr(teacher_model, 'forward_with_features'):
                    teacher_outputs, teacher_features = teacher_model.forward_with_features(clean_data)
                else:
                    teacher_outputs = teacher_model(clean_data)
                    teacher_features = None

            temp = distill_cfg['temp']
            kd = F.kl_div(
                F.log_softmax(outputs / temp, dim=1),
                F.softmax(teacher_outputs / temp, dim=1),
                reduction='batchmean',
            ) * (temp * temp)
            loss = loss + distill_cfg['alpha'] * kd

            if features is not None and teacher_features is not None:
                feat_mse = F.mse_loss(features, teacher_features.detach())
                loss = loss + distill_cfg['beta'] * feat_mse

        if center_criterion is not None and features is not None:
            loss = loss + args.center_loss_weight * center_criterion(features, target)
        if args.node_contrastive_weight > 0:
            node_ctr = cross_node_contrastive_loss(node_features, temperature=args.node_contrastive_temp)
            if node_ctr is not None:
                loss = loss + args.node_contrastive_weight * node_ctr
        if args.node_consistency_weight > 0:
            node_cons = reliability_weighted_consistency_loss(node_features, node_weights=node_weights)
            if node_cons is not None:
                loss = loss + args.node_consistency_weight * node_cons
    
        # 反向传播和优化
        optimizer.zero_grad()
        if center_optimizer is not None:
            center_optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if center_optimizer is not None:
            center_optimizer.step()
        
        # 累加损失
        train_loss += loss.item() * data.size(0)
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
       
        # 打印批次训练信息
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f',
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item())
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    accuracy = 100. * correct / total
    
    logging.info("-" * 50)
    logging.info(f'train epoch : {epoch}  train loss: {avg_loss:.4f}  accuracy: {accuracy:.4f}%')
    
    return accuracy


def val(model, val_loader, criterion):
    """验证函数，评估模型在验证集上的性能"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data, _, target = unpack_batch(batch)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            outputs = model(data)
            loss = criterion(outputs, target)
    
            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    # 计算平均损失和准确率
    avg_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(avg_loss)
    accuracy = 100. * correct / total
    
    logging.info(f'val loss: {avg_loss:.4f}  accuracy: {accuracy:.4f}%')
    
    return avg_loss, accuracy


def test(model, test_loader):
    """测试函数，评估模型在测试集上的性能并计算混淆矩阵"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, _, target = unpack_batch(batch)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算准确率
    accuracy = 100. * correct / total
    
    # 计算混淆矩阵
    matrix = confusion_matrix(all_targets, all_preds)
    
    # 计算评估指标
    OA = np.trace(matrix) / np.sum(matrix) if np.sum(matrix) > 0 else 0
    AA = np.diag(matrix) / np.sum(matrix, axis=1) if np.sum(matrix, axis=1).any() else 0
    AA_mean = np.mean(AA) if len(AA) > 0 else 0
    total = np.sum(matrix)
    sum_rows = np.sum(matrix, axis=1)
    sum_cols = np.sum(matrix, axis=0)
    expected = np.outer(sum_rows, sum_cols) / total if total > 0 else 0
    Kappa = (OA - np.trace(expected)/total) / (1 - np.trace(expected)/total) if (1 - np.trace(expected)/total) > 0 else 0
    
    logging.info(f'Test results: accuracy: {accuracy:.4f}%')
    logging.info(f'OA: {100. * OA:.2f}%  AA_mean: {100. * AA_mean:.2f}%  Kappa: {Kappa:.4f}')
    
    return AA, OA, AA_mean, Kappa, matrix


if __name__ == '__main__':
    # 初始化评估指标存储字典
    metrics_arr = {'OA': [], 'AA_mean': [], 'Kappa': [], 'AA': []}
    
    total_time = []
    distill_enabled = bool(args.distill_teacher_ckpt and args.distill_clean_datapath and args.arch == 'scheme5')
    if distill_enabled:
        logging.info('蒸馏模式启用: teacher_ckpt=%s clean_datapath=%s', args.distill_teacher_ckpt, args.distill_clean_datapath)
    else:
        logging.info('蒸馏模式关闭')

    all_data = get_data(args)
    clean_all_data = None
    if distill_enabled:
        clean_args = argparse.Namespace(**vars(args))
        clean_args.datapath = args.distill_clean_datapath
        clean_all_data = get_data(clean_args)

    for t in range(args.iterTime):
        '准备数据'
        logging.info(f'||--------------------- 迭代 {t+1}/{args.iterTime} 开始 seed: {args.seed}---------------------||')
        
        # 使用data_read.py中的get_dataloaders函数获取数据加载器

        if distill_enabled:
            train_loader, test_loader, val_loader = get_paired_dataloaders(all_data, clean_all_data, args)
        else:
            train_loader, test_loader, val_loader = get_dataloaders(all_data, args)
        args.seed+=1
        '初始化模型'
        if args.arch == 'scheme1':
            model = RadarInterferenceModel_Scheme1(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
        elif args.arch == 'scheme2':
            model = RadarInterferenceModel_Scheme2(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
        elif args.arch == 'scheme3':
            model = RadarInterferenceModel_Scheme3(
                fusion_type=args.fusion_type,
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
        elif args.arch == 'scheme4':
            model = RadarInterferenceModel_Scheme4(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
        elif args.arch == 'scheme5':
            model = RadarInterferenceModel_Scheme5(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout,
                use_denoise_stem=args.use_denoise_stem,
                use_tf_cross_attn=args.use_tf_cross_attn,
                use_node_co_attn=args.use_node_co_attn,
                use_node_reliability=args.use_node_reliability,
                domain_fusion_type=args.domain_fusion_type,
                cls_head=args.cls_head,
                cosine_scale=args.cosine_scale
            )
        else:
            raise ValueError(f"未知的架构: {args.arch}")

        if args.cuda:
            model = model.cuda()

        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            model.load_state_dict(ckpt, strict=True)
            logging.info('已加载预训练参数: %s', args.resume)

        teacher_model = None
        distill_cfg = None
        if distill_enabled:
            teacher_model = RadarInterferenceModel_Scheme5(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                num_nodes=args.num_nodes,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout,
                use_denoise_stem=args.teacher_use_denoise_stem,
                use_tf_cross_attn=False,
                use_node_co_attn=False,
                use_node_reliability=False,
                domain_fusion_type='gate',
                cls_head='linear',
                cosine_scale=args.cosine_scale
            )
            teacher_ckpt = torch.load(args.distill_teacher_ckpt, map_location='cpu')
            if isinstance(teacher_ckpt, dict) and 'state_dict' in teacher_ckpt:
                teacher_ckpt = teacher_ckpt['state_dict']
            teacher_model.load_state_dict(teacher_ckpt, strict=True)
            if args.cuda:
                teacher_model = teacher_model.cuda()
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)

            distill_cfg = {
                'alpha': args.distill_alpha,
                'beta': args.distill_beta,
                'temp': args.distill_temp,
            }
            logging.info('蒸馏教师已加载: %s', args.distill_teacher_ckpt)
        
        # 定义优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        criterion = build_criterion(args)
        center_criterion = None
        center_optimizer = None
        if args.center_loss_weight > 0 and args.arch == 'scheme5' and hasattr(model, 'forward_with_features'):
            center_criterion = CenterLoss(num_classes=args.num_classes, feat_dim=args.d_model)
            if args.cuda:
                center_criterion = center_criterion.cuda()
            center_optimizer = optim.SGD(center_criterion.parameters(), lr=args.center_lr)
        
        logging.info('||--------------------- 模型初始化完成 ---------------------||')
        
        '保存路径设置'
        if args.searchParm:
            args.save = os.path.join(args.logDir, f"{args.arch}-t-{t}-Time-{args.time}")
        else:
            args.save = os.path.join(args.logDir, f"{args.arch}-t-{t}-Time-{args.time}")
        
        os.makedirs(args.save, exist_ok=True)

        '训练'
        best_val_acc = -1.0
        best_model_path = None
        epochs_without_improve = 0
        last_epoch = args.startEpoch - 1
        model.train()
        for epoch in range(args.startEpoch, args.epochs):
            last_epoch = epoch
            tic3 = time.time()
            train_acc = train(
                epoch,
                model,
                train_loader,
                optimizer,
                criterion,
                center_criterion,
                center_optimizer,
                teacher_model,
                distill_cfg,
            )  # 训练模型
            toc3 = time.time()
            logging.info('training time: %f', (toc3 - tic3))
            
            # 验证模型
            val_loss, val_acc = val(model, val_loader, criterion)
            scheduler.step()
            
            # 保存最佳模型
            if best_model_path is None or val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(args.save, f'best_model_iter_{t}.pth')
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improve = 0
                logging.info(f'最佳模型已保存至 {best_model_path}，验证准确率: {best_val_acc:.4f}%')
            else:
                epochs_without_improve += 1
                if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
                    logging.info('Early stopping at epoch %d after %d validation plateaus', epoch, epochs_without_improve)
                    break

        '保存最终模型'
        final_model_path = os.path.join(args.save, f'final_model_epoch_{last_epoch + 1}_iter_{t}.pth')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f'最终模型已保存至 {final_model_path}')

        '测试'
        with torch.no_grad():
            if best_model_path is not None:
                model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                if args.cuda:
                    model = model.cuda()
                logging.info(f'测试阶段加载最佳模型: {best_model_path}')
            model.eval()
            
            tic4 = time.time()
            AA, OA, AA_mean, Kappa, matrix = test(model, test_loader)
            toc4 = time.time()
            time1 = (toc4 - tic4) / len(test_loader.dataset)
            total_time.append(time1)
            logging.info('测试时间: %f', (toc4 - tic4))

            # 存储指标
            metrics_arr['OA'].append(OA)
            metrics_arr['AA_mean'].append(AA_mean)
            metrics_arr['Kappa'].append(Kappa)
            metrics_arr['AA'].append(AA)
            
            # 记录日志
            logging.info(f'-迭代 {t + 1} OA: {100. * OA:.2f}%')
            logging.info(f'-迭代 {t + 1} AA_mean: {100. * AA_mean:.2f}%')
            # logging.info(f'-迭代 {t + 1} AA: {100. * AA:.2f}%')
            logging.info(f'-迭代 {t + 1} Kappa: {Kappa:.4f}')
            logging.info(f'-迭代 {t + 1}各类别准确率: \n{100. * AA}')
            logging.info(f'混淆矩阵: \n{matrix}')

        '清理内存'
        if args.cuda:
            torch.cuda.empty_cache()
        del model

    # 记录所有迭代的均值和方差
    logging.info('|| ---------------- 所有迭代的平均结果 ------------------- ||')
    logging.info(f"平均测试时间: {np.mean(total_time):.6f} 秒/样本")
    
    OA_mean = np.mean(metrics_arr['OA'])
    OA_std = np.std(metrics_arr['OA'])
    AA_mean_mean = np.mean(metrics_arr['AA_mean'])
    AA_mean_std = np.std(metrics_arr['AA_mean'])
    Kappa_mean = np.mean(metrics_arr['Kappa'])
    Kappa_std = np.std(metrics_arr['Kappa'])

    
    logging.info(f'OA 均值: {100. * OA_mean:.2f}%  标准差: {100. * OA_std:.2f}%')
    logging.info(f'AA_mean 均值: {100. * AA_mean_mean:.2f}%  标准差: {100. * AA_mean_std:.2f}%')
    logging.info(f'Kappa 均值: {Kappa_mean:.4f}  标准差: {Kappa_std:.4f}')
    logging.info(f' 各类别平均准确率: \n{100. * np.mean(metrics_arr ["AA"], axis=0)}')
    logging.info(f' 各类别准确率标准差: \n{100. * np.std(metrics_arr ["AA"], axis=0)}')
    
    
    logging.info('|| ---------------- 实验完成 ------------------- ||')
