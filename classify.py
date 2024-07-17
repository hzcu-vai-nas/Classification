import numpy as np
from pathlib import Path
import random
import torch
import torchvision
from torchvision import transforms
import tqdm
import argparse
import logging
import nni
from nni.utils import merge_parameter
import pydicom
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import os
import torch.nn.functional as F
import subprocess
import atexit
from nni.trial import get_trial_id

import pandas as pd

from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear

if os.path.exists('log.log'):
    os.remove('log.log')

if os.path.exists('image_urls.txt'):
    os.remove('image_urls.txt')    

os.mknod('image_urls.txt')
    
#if os.path.exists('result.log'):
#    os.remove('result.log')

def setup_logger(logger_name, file_name):
    # 创建 logger 对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器，并设置日志级别和文件名
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)

    # 将文件处理器添加到 logger 对象中
    logger.addHandler(file_handler)

    return logger

# 使用函数设置日志记录器
logger = setup_logger('infomation', 'log.log')
result_logger = setup_logger('result', 'result.log')
params_logger = setup_logger('hyperparams result', 'params.log')


# 数据预处理函数，用于处理JPG文件
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # 将图像转换为灰度图像
    image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
    image = image.astype(np.float32) / 255.0  # 将图像值标准化到 [0, 1] 范围内
    return transforms.ToTensor()(image)


# 数据集目录和转换
data_dir = Path("./data")

# 创建结果文件夹
result_dir = Path("./result")
result_dir.mkdir(parents=True, exist_ok=True)

# 创建数据集和数据加载器
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform=None, image_ext=".jpg"):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.image_ext = image_ext  # 添加一个变量来保存图像文件的后缀

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"{self.df.iloc[idx, 0]}{self.image_ext}")  # 在文件名后添加后缀
        image = preprocess_image(img_name)
        label = self.df.iloc[idx, 1]  # 获取对应的类别标签
        return image, label



# 配置超参数搜索空间
def get_params():
    parser = argparse.ArgumentParser(description='PyTorch Brain Tumor Example')

    parser.add_argument("--batch_size", type=int, default=4, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument("--hidden_size", type=int, default=64, metavar='N', help='hidden layer size (default: 64)')
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument("--epochs", type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument("--conv2", type=int, default=0, metavar='N', help='how to do with conv2 (default: 0)')
    parser.add_argument("--dropout", type=int, default=0.5, metavar='N', help='how to do with dropout (default: 0.5)')
    parser.add_argument("--feature", type=int, default=32, metavar='N', help='how to do with feature (default: 32)')

    parser.add_argument("--seed", type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--log_interval", type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument("--weights", type=str, default="./weights", help="folder to save weights")


    args, _ = parser.parse_known_args()
    return args



# 定义模型结构

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# NAS搜索模型
class MyModelSpace(ModelSpace):
    def __init__(self,conv2,dropout,feature):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        if(conv2 == 0):
            self.conv2  = nn.Conv2d(32, 64, 3, 1)
        else:
            self.conv2  = DepthwiseSeparableConv(32, 64)
        # nni.choice is used to select a dropout rate.
        # The result can be used as parameters of `MutableXXX`.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.5)
        feature = feature
        self.fc1 = MutableLinear(891136, feature)
        self.fc2 = MutableLinear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


# 存储训练过程状态
training_loss_history = []
validation_loss_history = []
validation_accuracy_history = []

# 定义训练函数
def train(args, model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_training_loss = []  # 初始化损失列表
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        epoch_training_loss.append(loss.item())  # 将每个batch的损失添加到当前epoch的损失列表中
    # 计算当前epoch的平均训练损失并添加到训练损失历史列表中
    epoch_avg_training_loss = np.mean(epoch_training_loss)
    training_loss_history.append(epoch_avg_training_loss)


# 定义测试和验证函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    validation_loss_history.append(test_loss)
    validation_accuracy = 100. * correct / len(test_loader.dataset)
    validation_accuracy_history.append(validation_accuracy)

    return validation_accuracy
    
# 主函数
def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    logger.info(device)

    training_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(args['hidden_size']),
        transforms.CenterCrop(args['hidden_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(args['hidden_size']),
        transforms.CenterCrop(args['hidden_size']),
        transforms.ToTensor(),
    ])

    # 创建训练和验证数据集
    training_dataset = CustomDataset(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=training_transform, image_ext=".jpg")
    validation_dataset = CustomDataset(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=validation_transform, image_ext=".jpg")


    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=True, **kwargs)

    # 创建基于NAS搜索的模型
    model = MyModelSpace(args['conv2'],args['dropout'],args['feature']).to(device)
    logger.info(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    best_acc = 0
    best_hyperparameters = None
    
    # 获取试验ID,用来分别输出每次试验的绘制图像
    exp_id = nni.get_experiment_id()
    id = nni.get_trial_id()
    # result_logger.info(f"The parameters result of trial: {id}")
    params_logger.info(f"The parameters result of trial: {id}")
    
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        nni.report_intermediate_result(test_acc)
        if test_acc > best_acc:
            best_acc =test_acc
            best_hyperparameters = (args['batch_size'],args['hidden_size'],args['lr'],args['momentum'])
    nni.report_final_result(test_acc)
    
    params_logger.info(f'Best hyperparameters：{best_hyperparameters}\n')
    
    
    # 绘制训练和验证损失
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(%)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(result_dir / f"training_validation_loss_{id}.png")  # 保存图形到结果文件夹
    plt.close()
    image_urls_input = result_dir / f"training_validation_loss_{id}.png"
    
    # 将图片路径写入到 image_urls.txt 文件中
    with open('image_urls.txt', 'a') as f:
        f.write(str(image_urls_input)+"\n")

    # 绘制验证精度图
    plt.plot(validation_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(result_dir / f"validation_accuracy_{id}.png")  # 保存图形到结果文件夹
    plt.close()
    
    image_urls_input = result_dir / f"validation_accuracy_{id}.png"
    
    # 将图片路径写入到 image_urls.txt 文件中
    with open('image_urls.txt', 'a') as f:
        f.write(str(image_urls_input)+"\n")
    

if __name__ == '__main__':
    try:
        # 其他代码部分
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
