import copy
import time
from torchvision.datasets import FashionMNIST#导入数据集
from torchvision import transforms#导入图像预处理工具包
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
import torch
import pandas as pd

#处理训练集和验证集，数据加载

#导入训练数据集和验证集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                              download=True)
    #训练集占80%，验证集占20%
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    #划分数据批次,验证集和数据集最好是一样的格式
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=2)
    value_loader = Data.DataLoader(dataset=val_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=2)
    return train_loader,value_loader

#模型训练函数
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    # 有gpu用gpu，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，定义一个优化器，进行参数更新，梯度下降法,学习率
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    #把模型放到device中
    model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #初始化参数
    #最高准确度
    best_acc=0.0
    #训练集损失函数列表
    train_loss_all=[]
    #验证集损失函数列表
    val_loss_all=[]
    #训练集精度列表
    train_acc_all=[]
    #验证机准确度列表
    val_acc_all=[]
    #保存每轮训练的时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)

        #初始化参数
        #训练集损失函数
        train_loss = 0.0
        #训练集精度
        train_corrects = 0.0
        #验证集损失函数
        val_loss = 0.0
        #验证集精度
        val_corrects = 0.0
        #训练集样本数量
        train_num = 0
        #测试集样本数量
        val_num = 0

        #开始训练
        for step,(b_x,b_y) in enumerate(train_dataloader):
            #把数据放到设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #把模型打开到训练模式
            model.train()
            #前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            out_put = model(b_x)
            #查找每一行中最大值对应的行标，得到每一行的样本的标签类别
            pre_lab=torch.argmax(out_put,dim=1)

            loss = criterion(out_put,b_y)
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()
            #根据反向传播得参数信息来更新网络的参数，起到降低loss函数计算值的作用
            optimizer.step()
            #对损失函数进行累加
            train_loss += loss.item()*b_x.size(0)
            #计算每个批次中正确预测的数量
            train_corrects += torch.sum(pre_lab==b_y)
            #计算到目前为止一共处理了多少样本
            train_num += b_x.size(0)

        #非测试集，是验证集。看看这一轮训练得怎么样
        for step,(b_x,b_y) in enumerate(val_dataloader):
            #把特征放到验证设备中
            b_x = b_x.to(device)
            #把标签放到验证设备中
            b_y = b_y.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            out_put = model(b_x)
            # 查找每一行中最大值对应的行标，得到每一行的样本的标签类别
            pre_lab = torch.argmax(out_put, dim=1)
            #计算损失函数
            loss = criterion(out_put,b_y)

            #不用进行反向传播更新参数的过程
            #但是可以看一下精度
            #对损失函数进行累加
            val_loss += loss.item()*b_x.size(0)
            #如果预测正确，则准确度加一
            val_corrects += torch.sum(pre_lab==b_y)
            #当前用于验证的样本数量
            val_num +=b_x.size(0)

         #测试集：计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects/train_num)
        #验证集
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects/val_num)

        #打印每一轮的结果
        print('{} train loss:{:.4f} train acc:{:.4f} '.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))

        #寻找最高准确度
        if val_acc_all[-1]>best_acc:
            #保存当前最高准确度
            best_acc = val_acc_all[-1]
            #保存当前模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        #该轮次训练花费的时间
        time_used = time.time() - since
        print("训练和验证耗费的时间{:.0f}m {:.0f}s：".format(time_used // 60, time_used % 60))

    #保存最优模型参数
    torch.save(best_model_wts,"../AlexNet/best_model.pth")
    train_process=pd.DataFrame(data={"epoch":range(len(train_loss_all)),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})

    return train_process

 #定义一个画图的函数
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    #图里面有几张图

    #一行两列的第一张图
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train_loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # 一行两列的第二张图
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val_acc")
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()

#从这里开始运行代码
if __name__ == '__main__':
    #将模型实例化
    AlexNet=AlexNet()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process=train_model_process(AlexNet,train_dataloader,val_dataloader,num_epochs=20)
    matplot_acc_loss(train_process)
