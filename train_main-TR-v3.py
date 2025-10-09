import argparse
import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
# 导入必要的模块
from model import NodeClassifier, RadarInterferenceModel_Scheme6
from data_read_v1 import get_dataloaders, get_data

# 全局变量
train_loss_list = []
val_loss_list = []

# 设置随机种子以确保可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_node_model(node_idx, model, train_loader, optimizer, criterion, epoch, args, device):
    """训练单个节点模型的批次训练函数"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 只使用当前节点的数据
        node_data = data[:, node_idx:node_idx+1, :, :, :]
        node_data, target = node_data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(node_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 累加损失和计算准确率
        train_loss += loss.item() * node_data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 打印批次训练信息
        if batch_idx % args.log_interval == 0:
            logging.info('Node %d: Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f',
                        node_idx, epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item())
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = 100. * correct / total
    
    logging.info(f'Node {node_idx}: train epoch : {epoch}  train loss: {avg_loss:.4f}  accuracy: {accuracy:.4f}%')
    return avg_loss, accuracy


def val_node_model(node_idx, model, val_loader, criterion, args, device):
    """验证单个节点模型的函数"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            node_data = data[:, node_idx:node_idx+1, :, :, :]
            node_data, target = node_data.to(device), target.to(device)
            
            output = model(node_data)
            loss = criterion(output, target)
            
            val_loss += loss.item() * node_data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = val_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total
    
    logging.info(f'Node {node_idx}: val loss: {avg_loss:.4f}  accuracy: {accuracy:.4f}%')
    return avg_loss, accuracy


def test(model, test_loader, device):
    """测试函数，返回模型输出和标签"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            all_logits.extend(output.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return np.array(all_logits), np.array(all_labels)


def get_accuracy(all_preds, all_labels):
    """计算评估指标"""
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算OA (Overall Accuracy)
    OA = accuracy_score(all_labels, all_preds)
    # 计算AA_mean (Average Accuracy)
    AA = np.diag(cm) / np.sum(cm, axis=1) if np.sum(cm, axis=1).any() else 0
    AA_mean = np.mean(np.diag(cm) / np.sum(cm, axis=1)) if np.sum(cm, axis=1).any() else 0
    # 计算Kappa系数
    total_samples = np.sum(cm)
    if total_samples == 0:
        return 0, 0, 0, 0, cm
    
    expected_accuracy = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total_samples ** 2)
    Kappa = (OA - expected_accuracy) / (1 - expected_accuracy) if expected_accuracy < 1 else 1.0
    
    return AA, OA, AA_mean, Kappa, cm


# 定义Scheme6组合模型类
class CombinedScheme6(nn.Module):
    def __init__(self, node_models):
        super(CombinedScheme6, self).__init__()
        self.node_models = nn.ModuleList(node_models)
    
    def forward(self, x):
        # 分别获取每个节点模型的输出
        outputs = []
        for i, model in enumerate(self.node_models):
            # 只使用对应节点的数据
            node_x = x[:, i:i+1, :, :, :]
            outputs.append(model(node_x))
        
        # 对输出进行堆叠
        stack_output = torch.stack(outputs, dim=1)
        return stack_output


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Radar Interference Classification - Scheme6 Training')
    
    # 数据设置
    parser.add_argument('--datapath', type=str, default=r'F:/Radar_data/DATA/12种干扰样式数据/', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='输入训练集的批量大小')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='输入测试集的批量大小')
    parser.add_argument('--num_classes', type=int, default=12, help='分类类别数')
    parser.add_argument('--train_size', type=float, default=100, help='训练集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    
    # 训练设置
    parser.add_argument('--iterTime', type=int, default=5, metavar='N', help='重复训练次数')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='训练轮数')
    parser.add_argument('--startEpoch', default=0, type=int, metavar='N', help='起始轮数')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='学习率')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='权重衰减')
    
    # 其他设置
    parser.add_argument('--noCuda', action='store_true', default=False, help='禁用CUDA训练')
    parser.add_argument('--cuda', type=bool, default=True, help='启用CUDA训练')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='日志记录间隔')
    parser.add_argument('--seed', type=int, default=10, help='随机种子')
    parser.add_argument('--logDir', default='./log-TR-v3', type=str, metavar='PATH', help='日志和模型保存路径')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=2, help='输入维度（实部+虚部）')
    parser.add_argument('--input_period', type=int, default=8, help='雷达周期数')
    parser.add_argument('--patch_size', type=int, default=16, help='Token分块尺寸')
    parser.add_argument('--stride', type=int, default=16, help='卷积步长')
    parser.add_argument('--seq_len', type=int, default=2000, help='单脉冲周期采样点')
    parser.add_argument('--d_model', type=int, default=128, help='特征维度')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='隐藏层维度倍率')
    parser.add_argument('--nhead', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_blocks', type=int, default=4, help='编码模块堆叠数')
    parser.add_argument('--dropout', type=float, default=0.0, help='丢弃率')
    
    args = parser.parse_args()
    args.cuda = not args.noCuda and torch.cuda.is_available()
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志路径
    args.time = time.strftime("%Y%m%d-%H%M%S")
    args.logTxtPath = os.path.join(args.logDir, f"scheme6-iterTime-{args.iterTime}-heads-{args.nhead}-lr-{args.lr}-epochs-{args.epochs}-Time-{args.time}-log.txt")
    
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
    logging.info('weight_decay %f', args.weight_decay)
    logging.info('d_model %d', args.d_model)
    logging.info('patch_size %d', args.patch_size)
    logging.info('num_blocks %d', args.num_blocks)
    logging.info('nhead %d', args.nhead)
    logging.info('mlp_ratio %d', args.mlp_ratio)
    logging.info('dropout %f', args.dropout)
    logging.info('architecture: scheme6')
    logging.info('Using device: %s', device)
    
    # 初始化评估指标存储字典
    metrics_arr = {
        'node_0': {'OA': [], 'AA_mean': [], 'Kappa': [], 'AA': []},
        'node_1': {'OA': [], 'AA_mean': [], 'Kappa': [], 'AA': []},
        'node_2': {'OA': [], 'AA_mean': [], 'Kappa': [], 'AA': []},
        'combined': {'OA': [], 'AA_mean': [], 'Kappa': [], 'AA': []}
    }
    
    total_time = []
    all_data = get_data(args)
    
    for t in range(args.iterTime):
        logging.info(f'||--------------------- 迭代 {t+1}/{args.iterTime} 开始 seed: {args.seed}---------------------||')
        
        # 获取数据加载器
        train_loader, test_loader, val_loader = get_dataloaders(all_data, args)
        args.seed += 1
        
        # 为每个节点训练单独的模型
        node_models = []
        for node_idx in range(3):  # 三个节点
            logging.info(f'||--------------------- 开始训练节点 {node_idx} ---------------------||')
            
            # 创建模型保存目录
            save_dir = os.path.join(args.logDir, f"scheme6-t-{t}-node-{node_idx}-Time-{args.time}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 创建节点分类器模型
            model = NodeClassifier(
                input_dim=args.input_dim,
                input_period=args.input_period,
                patch_size=args.patch_size,
                stride=args.stride,
                seq_len=args.seq_len,
                d_model=args.d_model,
                mlp_ratio=args.mlp_ratio,
                nhead=args.nhead,
                num_blocks=args.num_blocks,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
            model = model.to(device)
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss().to(device) if args.cuda else nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1, last_epoch=-1)
            
            best_val_acc = 0.0
            # 开始训练
            for epoch in range(args.startEpoch, args.epochs):
                tic = time.time()
                train_loss, train_acc = train_node_model(node_idx, model, train_loader, optimizer, criterion, epoch, args, device)
                toc = time.time()
                logging.info('Node %d: training time: %f', node_idx, (toc - tic))
                
                # 验证模型
                val_loss, val_acc = val_node_model(node_idx, model, val_loader, criterion, args, device)
                scheduler.step()
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(save_dir, f'best_model_node_{node_idx}_iter_{t}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f'Node {node_idx}: 最佳模型已保存至 {best_model_path}，验证准确率: {best_val_acc:.4f}%')
            
            # 保存最终模型
            final_model_path = os.path.join(save_dir, f'final_model_node_{node_idx}_epoch_{args.epochs}_iter_{t}.pth')
            torch.save(model.state_dict(), final_model_path)
            logging.info(f'Node {node_idx}: 最终模型已保存至 {final_model_path}')
            
            node_models.append(model)
        
        # 创建组合模型（对应Scheme6）
        combined_model = CombinedScheme6(node_models)
        combined_model = combined_model.to(device)
        
        # 在测试集上评估组合模型
        logging.info('||--------------------- 测试组合模型 Scheme6 ---------------------||')
        with torch.no_grad():
            tic = time.time()
            all_logits, all_labels = test(combined_model, test_loader, device)
            toc = time.time()
            test_time = toc - tic
            total_time.append(test_time / len(test_loader.dataset))
            logging.info('测试时间: %f', test_time)
        
        # 评估单个节点模型
        for node_idx in range(3):
            AA, OA, AA_mean, Kappa, cm = get_accuracy(
                np.argmax(all_logits[:, node_idx, :], axis=-1), 
                all_labels
            )
            metrics_arr[f'node_{node_idx}']['OA'].append(OA)
            metrics_arr[f'node_{node_idx}']['AA_mean'].append(AA_mean)
            metrics_arr[f'node_{node_idx}']['Kappa'].append(Kappa)
            metrics_arr[f'node_{node_idx}']['AA'].append(AA)
            
            logging.info(f'Node {node_idx} 测试结果: AA: \n{100. * AA}')
            logging.info(f'Node {node_idx} OA: {100. * OA:.2f}%  AA_mean: {100. * AA_mean:.2f}%  Kappa: {Kappa:.4f}')
            logging.info(f'Node {node_idx} 混淆矩阵: \n{cm}')
        
        # 评估组合模型
        AA, OA, AA_mean, Kappa, cm = get_accuracy(
            np.argmax(np.mean(all_logits, axis=1), axis=-1), 
            all_labels
        )
        metrics_arr['combined']['OA'].append(OA)
        metrics_arr['combined']['AA_mean'].append(AA_mean)
        metrics_arr['combined']['Kappa'].append(Kappa)
        metrics_arr['combined']['AA'].append(AA)
        
        logging.info(f'组合模型 Scheme6 测试结果: AA: \n{100. * AA}')
        logging.info(f'组合模型 Scheme6 OA: {100. * OA:.2f}%  AA_mean: {100. * AA_mean:.2f}%  Kappa: {Kappa:.4f}')
        logging.info(f'组合模型 Scheme6 混淆矩阵: \n{cm}')
        
        # 保存组合模型
        combined_save_dir = os.path.join(args.logDir, f"scheme6-t-{t}-Time-{args.time}")
        os.makedirs(combined_save_dir, exist_ok=True)
        combined_model_path = os.path.join(combined_save_dir, f'combined_scheme6_iter_{t}.pth')
        torch.save(combined_model.state_dict(), combined_model_path)
        logging.info(f'组合模型已保存至 {combined_model_path}')
        
        # 清理内存
        if args.cuda:
            torch.cuda.empty_cache()
        del model, combined_model
        logging.info(f'||--------------------- 迭代 {t+1}/{args.iterTime} 结束 ---------------------||')
    
    # 记录所有迭代的均值和方差
    logging.info('|| ---------------- 所有迭代的平均结果 ------------------- ||')
    logging.info(f"平均测试时间: {np.mean(total_time):.6f} 秒/样本")
    
    # 打印各节点和组合模型的平均指标
    for node_key in metrics_arr:
        OA_mean = np.mean(metrics_arr[node_key]['OA'])
        OA_std = np.std(metrics_arr[node_key]['OA'])
        AA_mean_mean = np.mean(metrics_arr[node_key]['AA_mean'])
        AA_mean_std = np.std(metrics_arr[node_key]['AA_mean'])
        Kappa_mean = np.mean(metrics_arr[node_key]['Kappa'])
        Kappa_std = np.std(metrics_arr[node_key]['Kappa'])
        
        logging.info(f'{node_key} OA 均值: {100. * OA_mean:.2f}%  标准差: {100. * OA_std:.2f}%')
        logging.info(f'{node_key} AA_mean 均值: {100. * AA_mean_mean:.2f}%  标准差: {100. * AA_mean_std:.2f}%')
        logging.info(f'{node_key} Kappa 均值: {Kappa_mean:.4f}  标准差: {Kappa_std:.4f}')
        logging.info(f' 各类别平均准确率: \n{100. * np.mean(metrics_arr[node_key]["AA"], axis=0)}')
        logging.info(f' 各类别准确率标准差: \n{100. * np.std(metrics_arr[node_key]["AA"], axis=0)}')
        
    
    logging.info('|| ---------------- 实验完成 ------------------- ||')


if __name__ == '__main__':
    main()