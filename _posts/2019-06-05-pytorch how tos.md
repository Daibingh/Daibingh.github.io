---
layout: post
title:  "pytorch How tos"
categories: pytorch
tags: pytorch DL
author: hdb
comments: true
excerpt: "with pytorch, how to do something common?"
---

* content
{:toc}

## 冻结参数

```py
for param in model.parameters():
    param.requires_grad = False
```

## 获取模型参数个数

```py
def get_params_num(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数
    total_num = sum(p.numel() for p in model.parameters())  # 全部参数
    return trainable_num, total_num
```

## 节省 GPU 内存技巧

- 尽量使用 in-place 操作：
    - Relu(inplace=True)
    - fill_() zero_() normal_() uniform_() exponential_()
- 变量内存重用
    - 网络输入和 label 等相对静态的变量尽量重用内存
- 不同的变量尽量覆盖
- 变量随用随计算
- SGD 比 Adam 在反向传播时更省显存

## 使用数据加载器

要使用 pytorch 的数据加载器，分两个主要步骤：
- 继承并实现 torch.utils.data.Dataset 类, 该类用于索取单个训练样本的数据，数据预处理的步骤也在此完成
- 使用 torch.utils.data.DataLoader 类, 该类实现批量、并行、随机加载数据

示例
```py
import torch
from torch.utils.data import Dataset, DataLoader

class MnistDataset(Dataset):

    def __init__(self, file, x_key, y_key):
        with open(file, 'rb') as f:
            data = pk.load(f)
        x = data.get(x_key)
        self.image_mean = np.mean(x)
        self.image_std = np.std(x)
        self.y = data.get(y_key)
        self.images = x.reshape(-1, 1, 28, 28)
        del data

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return {"image": image, "y": y}
        
dataset = MnistDataset('/home/hdb/py-pro/kaggle/mnist/data.pkl', 'x_train', 'y_train')
loader = DataLoader(dataset, batch_size, num_workers=0, shuffle=True)

for _, data in enumerate(loader):
    x_batch = data.get('image').to(torch.device("cuda"))
    y_batch = data.get('y').to(torch.device("cuda"))
    ...
```

## 自定义模型

继承并实现 nn.Module

```py
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(10, 10))
        self.fc.append(nn.Linear(10, 50))
        self.fc.append(nn.Linear(50, 10))
        self.fc.append(nn.Linear(10, 10))
        self.fc.append(nn.Linear(10, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < 4:
                x = torch.sigmoid(x)
        return x
```

## 模型保存和恢复

```py
# save model weights
torch.save(model.state_dict(), PATH)

# load model weights
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
```

## 设置学习率衰减

```py
# call the function in training loop
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 3:
        lr = .002
    elif epoch < 6:
        lr = .001
    elif epoch<8:
        lr = .0005
    else:
        lr = .0002
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

## 清空 GPU 内存

```py
torch.cuda.empty_cache()
```
