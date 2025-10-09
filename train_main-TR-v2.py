import argparse
import sys
import time
import os
from sklearn.metrics import confusion_matrix
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader

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
parser.add_argument('--dropout', type=float, default=0.0, help='dropout率')
parser.add_argument('--fusion_type', type=str, default='weighted', choices=['weighted', 'average'], help='节点融合类型')

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
logging.info('architecture: %s', args.arch)
logging.info('fusion_type: %s', args.fusion_type)

# 全局变量
train_loss_list = []
val_loss_list = []

def train(epoch, model, train_loader, optimizer, criterion):
    """训练函数"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # 调整数据维度以匹配模型期望的输入格式
        # 输入应形如: [batch_size, num_nodes, seq_len, input_dim, num_periods]
        # 假设当前data已经是正确的维度，无需额外处理
        
        # 获取模型输出
        outputs = model(data)
        
        # 计算损失
        loss = criterion(outputs, target)
    
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
        for batch_idx, (data, target) in enumerate(val_loader):
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
        for batch_idx, (data, target) in enumerate(test_loader):
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
    all_data = get_data(args)
    for t in range(args.iterTime):
        '准备数据'
        logging.info(f'||--------------------- 迭代 {t+1}/{args.iterTime} 开始 seed: {args.seed}---------------------||')
        
        # 使用data_read.py中的get_dataloaders函数获取数据加载器
        
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
                dropout=args.dropout
            )
        else:
            raise ValueError(f"未知的架构: {args.arch}")

        if args.cuda:
            model = model.cuda()
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1, last_epoch=-1)
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.cuda else torch.nn.CrossEntropyLoss()
        
        logging.info('||--------------------- 模型初始化完成 ---------------------||')
        
        '保存路径设置'
        if args.searchParm:
            args.save = os.path.join(args.logDir, f"{args.arch}-t-{t}-Time-{args.time}")
        else:
            args.save = os.path.join(args.logDir, f"{args.arch}-t-{t}-Time-{args.time}")
        
        os.makedirs(args.save, exist_ok=True)

        '训练'
        best_val_acc = 0.0
        model.train()
        for epoch in range(args.startEpoch, args.epochs):
            tic3 = time.time()
            train_acc = train(epoch, model, train_loader, optimizer, criterion)  # 训练模型
            toc3 = time.time()
            logging.info('training time: %f', (toc3 - tic3))
            
            # 验证模型
            val_loss, val_acc = val(model, val_loader, criterion)
            scheduler.step()
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(args.save, f'best_model_iter_{t}.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f'最佳模型已保存至 {best_model_path}，验证准确率: {best_val_acc:.4f}%')

        '保存最终模型'
        final_model_path = os.path.join(args.save, f'final_model_epoch_{args.epochs}_iter_{t}.pth')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f'最终模型已保存至 {final_model_path}')

        '测试'
        with torch.no_grad():
            # 加载最佳模型进行测试
            # model.load_state_dict(torch.load(best_model_path))
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