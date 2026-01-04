# 导入操作系统接口模块，用于处理文件和路径
import os
# 导入CLIP模型库
import clip
# 导入PyTorch深度学习框架
import torch

# 导入NumPy数值计算库
import numpy as np
# 从scikit-learn导入逻辑回归分类器
from sklearn.linear_model import LogisticRegression
# 从PyTorch导入数据加载器，用于批量加载数据
from torch.utils.data import DataLoader
# 导入CIFAR100数据集
from torchvision.datasets import CIFAR100
# 导入tqdm进度条库，用于显示处理进度
from tqdm import tqdm

# 加载模型
# 检查是否有可用的GPU，有则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载CLIP模型和预处理函数
# 'ViT-B/32'表示使用Vision Transformer Base模型，patch大小为32x32
model, preprocess = clip.load('ViT-B/32', device)

# 加载数据集
# 设置数据集保存路径为用户主目录下的.cache文件夹
root = os.path.expanduser("~/.cache")
# 加载CIFAR100训练集，应用CLIP的预处理转换
# download=True: 如果数据集不存在则自动下载
# train=True: 加载训练集（5万张图片）
# transform=preprocess: 对每张图片应用CLIP的预处理函数
train = CIFAR100(root, download=True, train=True, transform=preprocess)
# 加载CIFAR100测试集，同样应用CLIP的预处理
# train=False: 加载测试集（1万张图片）
test = CIFAR100(root, download=True, train=False, transform=preprocess)


# 定义函数：提取数据集的特征向量
def get_features(dataset):
    """提取数据集中所有图像的特征向量

    参数：
    dataset: PyTorch Dataset对象，包含图像和标签

    返回：
    all_features: NumPy数组，形状为[N, 512]，N是数据集大小，512是特征维度
    all_labels: NumPy数组，形状为[N]，对应的标签
    """
    # 存储所有特征向量的列表
    all_features = []
    # 存储所有标签的列表
    all_labels = []

    # 禁用梯度计算，节省内存和计算资源（推理阶段不需要梯度）
    with torch.no_grad():
        # 使用DataLoader批量加载数据，batch_size=100，tqdm显示进度条
        # 示例：CIFAR100训练集有50000张图片，将分500批处理
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            # 将图像批量转移到指定设备（GPU/CPU）
            # 提取图像特征向量
            # images形状: [100, 3, 224, 224] (批次大小100, 3通道, 224x224像素)
            # features形状: [100, 512] (批次大小100, 512维特征)
            features = model.encode_image(images.to(device))

            # 将当前批次的特征和标签添加到列表中
            all_features.append(features)
            all_labels.append(labels)

    # 将列表中所有批次的特征连接起来，转换为NumPy数组
    # torch.cat(all_features): 将500个[100,512]张量连接成[50000,512]张量
    # .cpu().numpy(): 将张量从GPU转移到CPU，并转换为NumPy数组
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# 计算训练集图像特征
# 输入：train数据集（5万张预处理后的图像）
# 输出：train_features形状[50000,512], train_labels形状[50000]
train_features, train_labels = get_features(train)
# 计算测试集图像特征
# 输入：test数据集（1万张预处理后的图像）
# 输出：test_features形状[10000,512], test_labels形状[10000]
test_features, test_labels = get_features(test)

# 执行逻辑回归分类
# 创建逻辑回归分类器
# random_state=0: 设置随机种子，确保结果可复现
# C=0.316: 正则化强度的倒数，较小的C表示更强的正则化
# max_iter=1000: 最大迭代次数，确保模型收敛
# verbose=1: 显示训练过程日志
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
# 在训练特征上训练分类器
# train_features: [50000, 512]特征矩阵
# train_labels: [50000]标签向量
classifier.fit(train_features, train_labels)

# 使用逻辑回归分类器进行评估
# 在测试特征上进行预测
# test_features: [10000, 512]测试特征矩阵
# predictions: [10000]预测标签向量
predictions = classifier.predict(test_features)
# 计算准确率
# 1. test_labels == predictions: 比较真实标签和预测标签，得到布尔数组
# 2. .astype(float): 将布尔值转换为浮点数（True=1.0, False=0.0）
# 3. np.mean(): 计算平均值，即准确率
# 4. * 100.: 转换为百分比
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
# 打印准确率，保留3位小数
print(f"Accuracy = {accuracy:.3f}")