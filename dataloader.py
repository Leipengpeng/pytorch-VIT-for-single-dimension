"""
    覆写dataloader

    输入:
        root_addr, label_num, sampling_ratio[X,X,X], rand_seed, batch_size
    输出:
        train_loader, val_loader, test_loader

    底部包含测试代码
"""
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


def get_addr(root_addr, label_num, usage_type='train'):
    datas_addr = os.path.join(root_addr, f'X_{usage_type}_{label_num}Class.npy')
    labels_addr = os.path.join(root_addr, f'Y_{usage_type}_{label_num}Class.npy')
    return datas_addr, labels_addr


def increase_func(datas, increase_rate=500):
    return datas * increase_rate


def read_data(root_addr, label_num, usage_type, pre_process=increase_func):
    datas_addr, labels_addr = get_addr(root_addr, label_num, usage_type)
    temp_datas, labels = np.load(datas_addr), np.load(labels_addr)
    datas_length = len(temp_datas)

    datas = []
    for data in temp_datas:
        datas.append(data)
    datas = np.array(datas)
    datas = pre_process(datas)

    datas = torch.from_numpy(datas).float()
    labels = torch.tensor(labels, dtype=torch.long)

    return datas_length, datas, labels


class get_dataset(Dataset):
    def __init__(self, root_addr, label_num, usage_type):
        self.datas_length, self.datas, self.labels = read_data(root_addr, label_num, usage_type)

    def __len__(self):
        return self.datas_length

    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        return data, label


def rand_segmentation_double(dataset, sampling_ratio, rand_seed):
    torch.manual_seed(rand_seed)
    datas_length = dataset.__len__()
    size_1 = int(datas_length * sampling_ratio[0])
    size_2 = datas_length - size_1

    data_1, data_2 = torch.utils.data.random_split(
        dataset, [size_1, size_2]
    )
    return data_1, data_2


def rand_segmentation_triple(dataset, sampling_ratio, rand_seed):
    torch.manual_seed(rand_seed)
    datas_length = dataset.__len__()
    size_1 = int(datas_length * sampling_ratio[0])
    size_2 = int(datas_length * sampling_ratio[1])
    size_3 = datas_length - size_1 - size_2

    data_1, data_2, data_3 = torch.utils.data.random_split(
        dataset, [size_1, size_2, size_3]
    )
    return data_1, data_2, data_3


# 无分类情况
def get_dataloader(root_addr, label_num, batch_size, usage_type):
    dataset = get_dataset(root_addr, label_num, usage_type)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


# 二分类情况
def get_double_dataloader(root_addr, label_num, sampling_ratio, rand_seed, batch_size, usage_type):
    dataset = get_dataset(root_addr, label_num, usage_type)
    data_1, data_2 = rand_segmentation_double(dataset, sampling_ratio, rand_seed)

    loader_1 = DataLoader(data_1, batch_size=batch_size, shuffle=True)
    loader_2 = DataLoader(data_2, batch_size=batch_size, shuffle=True)

    return loader_1, loader_2


# 三分类情况
def get_triple_dataloader(root_addr, label_num, sampling_ratio, rand_seed, batch_size, usage_type):
    dataset = get_dataset(root_addr, label_num, usage_type)
    data_1, data_2, data_3 = rand_segmentation_triple(dataset, sampling_ratio, rand_seed)

    loader_1 = DataLoader(data_1, batch_size=batch_size, shuffle=True)
    loader_2 = DataLoader(data_2, batch_size=batch_size, shuffle=True)
    loader_3 = DataLoader(data_3, batch_size=batch_size, shuffle=True)

    return loader_1, loader_2, loader_3


def draw(data):
    data = data.numpy()
    time = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    plt.plot(time, np.abs(data))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('IQ Waveform')
    plt.grid(True)
    plt.show()


# 测试符合全覆盖原则
def test():
    root_addr = "../../dataset/FS-SEI_4800"
    label_num = 10
    batch_size = 32
    usage_type = 'test'
    loader = get_dataloader(root_addr, label_num, batch_size, usage_type)
    print(len(loader))
    print(type(loader))

    sampling_ratio = [0.8, 0.2]
    rand_seed = 14
    loader_1, loader_2 = get_double_dataloader(root_addr, label_num, sampling_ratio, rand_seed, batch_size, usage_type)
    print(len(loader_1))
    print(len(loader_2))

    sampling_ratio = [0.8, 0.1, 0.1]
    loader_1, loader_2, loader_3 = get_triple_dataloader(root_addr, label_num, sampling_ratio, rand_seed, batch_size,
                                                         usage_type)
    print(len(loader_1))
    print(len(loader_2))
    print(len(loader_3))

    for data, train_labels in loader_3:
        print(data.shape)
        print(train_labels.shape)
        draw(data[0])


if __name__ == '__main__':
    test()
