import scipy.io as sio
import os
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RadarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data(args):
    file_names = os.listdir(args.datapath)
    all_data = []
    all_labels = []

    for idx, file_name in enumerate(file_names):
        print(f"Processing file {idx+1}/{len(file_names)}: {file_name}")
        file_path = os.path.join(args.datapath, file_name)
        
        # 使用h5py读取MATLAB v7.3文件
        try:
            with h5py.File(file_path, 'r') as f:
                mat_data = np.array(f['data'])
                # h5py读取的数组维度可能与scipy.io不同，需要调整
                if mat_data.ndim == 4 and mat_data.shape[3] == 1:
                    mat_data = mat_data[:, :, :, 0]
        except Exception:
            # 如果不是v7.3格式，尝试用scipy.io.loadmat
            try:
                mat_data = sio.loadmat(file_path)['data']
            except Exception as e:
                print(f"读取文件{file_name}失败: {e}")
                continue
        
        
        # 拆分实部和虚部，形状变为2*3*1000*16000
        # real_part = np.real(mat_data)
        # imag_part = np.imag(mat_data)
        flat_data = np.array([x[0] for x in mat_data.flat], dtype=np.float64)
        real_part = flat_data.reshape(mat_data.shape)
        flat_data = np.array([x[1] for x in mat_data.flat], dtype=np.float64)
        imag_part =  flat_data.reshape(mat_data.shape)

        combined_data = np.stack((real_part, imag_part), axis=0)
        
        # 按2000为周期拆分16000个采样点，得到8个周期，形状变为2*3*1000*8*2000
        period_splits = np.split(combined_data, 8, axis=1)  #2 * 8*2000*1000*3
        period_data = np.array(period_splits)
        period_data = period_data.transpose(0, 3, 1, 4, 2)  # 调整为8*1000*2*3*2000
        all_data.append(period_data)
        all_labels.append(np.ones(period_data.shape[1]) * idx)  # 标签为文件索引    
        
    stacked_data = np.stack(all_data, axis=0)  # 在 axis=0 堆叠
    return stacked_data
   
def load_data(all_data, args):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for idx in range(all_data.shape[0]): 
        period_data = all_data[idx]  # 形状为8*1000*2*3*2000              
        sample_indices = np.array(range(period_data.shape[1]))
        np.random.seed(args.seed)
        np.random.shuffle(sample_indices)
        # 按照配置采样数量或比例提取样本
        if args.train_size<1.0:
            sample_num = int(period_data.shape[1] * args.train_size)         
        elif args.train_size>1.0:
            sample_num = int(args.train_size)
        
        train_indices = sample_indices[:sample_num]
        test_indices = sample_indices[sample_num:]    
            
        train_data = period_data[:, train_indices]
        test_data = period_data[:, test_indices]            
        # all_data.append(period_data)
        # all_labels.append(np.ones(period_data.shape[1]) * idx)  # 标签为文件索引
        
        X_train.append(train_data)
        y_train.append(np.ones(train_data.shape[1]) * idx)  # 标签为文件索引
        X_test.append(test_data)
        y_test.append(np.ones(test_data.shape[1]) * idx)  # 标签为文件索引
    
    X_train = np.concatenate(X_train, axis=1).transpose(1, 3, 4, 2, 0) 
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=1).transpose(1, 3, 4, 2, 0) 
    y_test = np.concatenate(y_test, axis=0)
    
    # all_data = np.concatenate(all_data, axis=1)
    # all_labels = np.concatenate(all_labels, axis=0)

    # # 调整维度为(N, 3, 2000, 2, 8)（N为样本数）
    # all_data = all_data.transpose(1, 3, 4, 2, 0)  # 形状为(1000, 3, 2000, 2, 8)等，根据采样后情况

    # # 划分训练集、测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     all_data, all_labels, test_size=config['test_size'], random_state=config['random_seed']
    # )
    # 从测试集划分验证集
    _, X_val, _, y_val = train_test_split(
        X_test, y_test, test_size=0.1, random_state=args.seed
    )

    # 转换为PyTorch张量
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def get_dataloaders(all_data, args):
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data(all_data, args)
    
    train_dataset = RadarDataset(X_train, y_train)
    test_dataset = RadarDataset(X_test, y_test)
    val_dataset = RadarDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    return train_loader, test_loader, val_loader