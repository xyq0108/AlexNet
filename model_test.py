import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet


#读取数据
def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.ToTensor()]),#转换成张量形式
                              download=True)
    #一张一张测试，之前是一个batch训练一次
    test_loader = Data.DataLoader(dataset=test_data,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)

    return test_loader

def test_model_process(model,train_dataloader):
    #定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #把模型放到设备当中
    model.to(device)
    #初始化参数
    test_acc=0.0
    test_num=0
    test_corrects=0.0
    #模型进行推理，不涉及反向传播，只进行前向传播，约定俗成的代码，不需要进行梯度计算
    with torch.no_grad():
        #获取数据，x是特征，y是标签
        for test_data_x,test_data_y in train_dataloader:
            #将特征放到设备当中
            test_data_x = test_data_x.to(device)
            #将标签放到设备当中
            test_data_y = test_data_y.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            #查找每一行中最大值对应的行标,若直接找最大值，而不转换成概率，就不知道最大值预测的准确度
            pre_lab=torch.argmax(output,dim=1)
            test_corrects += torch.sum(pre_lab==test_data_y)
            #将所有的测试样本进行累加
            test_num=test_num+1

    #计算测试准确率
    test_acc=test_corrects/test_num
    print("test_acc:{:.4f}".format(test_acc))


if __name__ == '__main__':
    #加载模型
    model=AlexNet()
    #将已经训练好的参数加载进来
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    #加载测试数据
    test_dataloader=test_data_process()
    test_model_process(model,test_dataloader)

    #看单张的图片测试出来的结果是什么
    #设置测试所要用到的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #把模型放到设备中
    model=model.to(device)

    classes=['T-shirt-top','trouser','pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #将模型设置为测试模型
            model.eval()
            output = model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：",classes[result],"----------","真实值",classes[label])

