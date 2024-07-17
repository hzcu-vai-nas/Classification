import numpy as np
from pathlib import Path
import random
import time
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
from nni.experiment import Experiment
import json
from flask import Flask, render_template, jsonify
import threading

import pandas as pd

from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear


app = Flask(__name__)

lock = threading.Lock()

model_dict = {}

@app.route('/')
def index():
    return render_template('index.html')
    

# 在一个新线程中运行 Flask 应用程序的函数
def run_flask_app():
    app.run(debug=True, use_reloader=False)


# 在程序退出时自动关闭端口为8081的NNI实验
def close_nni_experiment():
    subprocess.run(['nnictl', 'stop', '-p', '8090'])

    
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


# 数据集目录和转换
data_dir = Path("./data")

# 创建结果文件夹
result_dir = Path("./result")
result_dir.mkdir(parents=True, exist_ok=True)


# 数据预处理函数，用于处理JPG文件
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # 将图像转换为灰度图像
    image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
    image = image.astype(np.float32) / 255.0  # 将图像值标准化到 [0, 1] 范围内
    return transforms.ToTensor()(image)



# 创建数据集和数据加载器
class CustomDataset_csv(torch.utils.data.Dataset):
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
        label = self.df.iloc[idx, 1]  # 获取对应的类别标签，如果是全有肿瘤的数据集，直接设置为1即可
        return image, label


class CustomDataset_no_csv(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, image_ext=".jpg"):
        self.data_dir = data_dir
        self.transform = transform
        self.image_ext = image_ext

        self.image_paths = []
        self.labels = []

        # Iterate through all subdirectories
        for label, folder_name in enumerate(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                # Exclude the "notomor" folder for tumor images
                if folder_name != "notumor":
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith(image_ext):
                            img_path = os.path.join(folder_path, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(1)  # Label 1 for tumor
                else:
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith(image_ext):
                            img_path = os.path.join(folder_path, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(0)  # Label 0 for no tumor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = preprocess_image(img_path)
        label = self.labels[idx]
        return image, label


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MyModelSpace(ModelSpace):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ], label='conv2')
        # nni.choice is used to select a dropout rate.
        # The result can be used as parameters of `MutableXXX`.
        self.dropout1 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        feature = nni.choice('feature', [64, 128, 256])
        self.fc1 = MutableLinear(891136, feature)
        self.fc2 = MutableLinear(feature, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = torch.sigmoid(x)  # 使用 sigmoid 函数作为激活函数
        return output


model_space = MyModelSpace()
model_space
result_logger.info(model_space)

search_strategy = strategy.Random()

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy



def run_classification_task(model_path, data_dir):
    # Load the model
    model = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    
    labels = []
    # Load CSV file to get labels
    # csv_file = data_dir / "Brain Tumor.csv"
    # df = pd.read_csv(csv_file)
    # labels = df["Class"].tolist()  # Assuming the column name for labels is "label"
    
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Prepare new dataset
    # Assuming you have a function to load your new dataset, e.g., load_new_dataset()
    if os.path.exists(os.path.join(data_dir, "Brain Tumor.csv")):
        dataset = CustomDataset_csv(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=transf, image_ext=".jpg")
    else:
        # Use CustomDataset_no_csv if CSV file does not exist
        dataset = CustomDataset_no_csv(data_dir / "Training", transform=transf, image_ext=".jpg")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


    # result_logger.info(f"labels: {labels}\n\n\n\n")
    
    # Run classification on the new dataset
    results = []
    mismatches = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, axis=1)
            results.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())           

    for idx, (pred, true) in enumerate(zip(results, labels)):
        if pred != true:
            mismatches.append((idx, pred, true))
            
            
    result_logger.info(f"results: {results}")
    result_logger.info(f"There is {len(mismatches)} datas are mismatched!")
    return mismatches

    

def main():
    
    # 运行寻找模型的最佳超参的nni工作
    evaluator = FunctionalEvaluator(evaluate_model)
    exp = NasExperiment(model_space, evaluator, search_strategy)

    exp.config.max_trial_number = 10   # spawn 3 trials at most
    exp.config.trial_concurrency = 1  # will run 1 trial concurrently
    exp.config.trial_gpu_number = 0   # will not use GPU
    exp.config.training_service.use_active_gpu = False
    exp.run(port=8080)
    
    for model_dict in exp.export_top_models(formatter='dict'):
        result_logger.info(f"Best model parameters: {model_dict}") 
    with nni.nas.space.model_context(model_dict):
        final_model = MyModelSpace()
        torch.save(final_model , os.path.join("./weights", "trained_model.pt"))
        
    
    search_space = {
        "conv2": {"_type":"choice", "_value": [model_dict["conv2"]]},
        "dropout":{"_type":"choice", "_value": [model_dict["dropout"]]}, 
        "feature": {"_type":"choice", "_value": [model_dict["feature"]]}, 
        "batch_size": {"_type":"choice", "_value": [2,4,6,8,16]},
        "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
        "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
        "momentum":{"_type":"uniform","_value":[0, 1]}
    }


    classify_experiment = Experiment('local')
    classify_experiment.config.trial_command = 'python classify.py'
    classify_experiment.config.trial_code_directory = '.'
    classify_experiment.config.search_space = search_space
    classify_experiment.config.tuner.name = 'TPE'
    classify_experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    classify_experiment.config.max_trial_number = 10
    classify_experiment.config.trial_concurrency = 1

    classify_experiment.run(8090)
    
    
    # 获取所有试验的指标信息
    all_trials_metrics = classify_experiment.get_job_metrics()

    # 比较指标获取最佳试验 ID 和超参数
    best_trial_id = None
    best_metric_value = float('-inf')  # 初始化为负无穷
    best_hyperparameters = None

    for trial_id, metrics in all_trials_metrics.items():
        # 访问每个试验的指标数据
        default_metric_value = metrics[-1].data  # 获取最后一个指标数据（FINAL）
        if default_metric_value > best_metric_value:
            best_metric_value = default_metric_value
            best_trial_id = trial_id
            
    # 获取最佳超参数
            
    trial_info = classify_experiment.get_trial_job(best_trial_id)
    
    hyper_params_list = getattr(trial_info,'hyperParameters',None)
    print(hyper_params_list)
    
    if hyper_params_list:  # 确保 hyper_params_list 不为 None
        if isinstance(hyper_params_list, list):
            hyper_params_dict = {}  # 创建一个空字典
            # 获取 parameters 字典
            parameters_dict = hyper_params_list[0].parameters
            hyper_params_dict.update(parameters_dict)  # 直接更新字典
            print(hyper_params_dict)  # 输出转换后的字典
            result_logger.info(hyper_params_dict)
        else:
            print("Error: hyperParameters attribute is not a list.")
            # 处理 hyperParameters 不是列表的情况
    else:
        print("Error: hyperParameters attribute not found in trial_info.")
        # 处理 hyperParameters 属性未找到的逻辑

    # 输出最佳试验 ID 和超参数
    print("Best trial ID:", best_trial_id)
    print("Best hyperparameters:", hyper_params_dict)
    
    # 对导入了最佳参数的最佳模型进行训练
    
    train_final_model(final_model,hyper_params_dict)
    
    torch.save(final_model , os.path.join("./weights", "trained_model.pt"))
    
    # 对训练效果进行测试，并且输出
    
    mismatches = []
    model_path = "./weights/trained_model.pt"
    mismatches = run_classification_task(model_path, data_dir)
    
    result_logger.info(mismatches)
    
    
    
    
def train_final_model(model,params):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'],  weight_decay=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if os.path.exists(os.path.join(data_dir, "Brain Tumor.csv")):
        training_dataset = CustomDataset_csv(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=transf, image_ext=".jpg")
        validation_dataset = CustomDataset_csv(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=transf, image_ext=".jpg")
    else:
        training_dataset = CustomDataset_no_csv(data_dir / "Training", transform=transf, image_ext=".jpg")
        validation_dataset = CustomDataset_no_csv(data_dir / "Testing", transform=transf, image_ext=".jpg")

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)

    for epoch in range(6):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        

def evaluate_model(model):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,  weight_decay=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if os.path.exists(os.path.join(data_dir, "Brain Tumor.csv")):
        training_dataset = CustomDataset_csv(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=transf, image_ext=".jpg")
        validation_dataset = CustomDataset_csv(data_dir / "Brain Tumor", data_dir / "Brain Tumor.csv", transform=transf, image_ext=".jpg")
    else:
        training_dataset = CustomDataset_no_csv(data_dir / "Training", transform=transf, image_ext=".jpg")
        validation_dataset = CustomDataset_no_csv(data_dir / "Testing", transform=transf, image_ext=".jpg")

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, shuffle=True)

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)
    
if __name__ == '__main__':
    try:   
        global value_epoch
        
        flask_thread = threading.Thread(target=run_flask_app)
        flask_thread.start()
        
        main()       
        # 等待 Flask 应用程序线程结束
        flask_thread.join()
        
    except Exception as exception:
        logger.exception(exception)
        raise
