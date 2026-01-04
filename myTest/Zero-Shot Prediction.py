# 导入操作系统接口模块，用于文件路径操作
import os
# 导入OpenAI的CLIP库，用于多模态图像-文本模型
import clip
# 导入PyTorch深度学习框架
import torch
# 导入CIFAR-100数据集类，用于获取测试数据
from torchvision.datasets import CIFAR100

# 加载CLIP模型
# 检测GPU是否可用，优先使用GPU加速
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载ViT-B/32模型和对应的预处理函数
# ViT-B/32: Vision Transformer Base模型，patch大小32x32
model, preprocess = clip.load('ViT-B/32', device)

# 下载并加载CIFAR-100数据集
# root=os.path.expanduser("~/.cache"): 指定数据集缓存路径，~表示用户主目录
# download=True: 如果数据集不存在则自动下载
# train=False: 加载测试集（而不是训练集）
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# 准备输入数据
# 获取CIFAR-100测试集中的第3637个样本
# image: PIL图像对象，形状32x32
# class_id: 类别标签，0-99之间的整数
image, class_id = cifar100[3637]

# 预处理图像并添加批次维度
# preprocess(image): 将32x32图像转换为224x224，归一化等
# unsqueeze(0): 在第0维度添加批次维度，从[3,224,224]变为[1,3,224,224]
# to(device): 将图像张量移动到指定设备（GPU/CPU）
image_input = preprocess(image).unsqueeze(0).to(device)

# 为CIFAR-100的所有类别生成文本描述
# cifar100.classes: 包含100个类别名称的列表
# clip.tokenize(f"a photo of a {c}"): 为每个类别生成token
# torch.cat(): 将100个token张量连接成一个张量
# to(device): 将文本张量移动到指定设备
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# 计算图像和文本特征
# torch.no_grad(): 禁用梯度计算，节省内存（推理模式）
with torch.no_grad():
    # 提取图像特征
    # model.encode_image(): 使用CLIP图像编码器提取特征
    # 输出形状: [1, 512] (1张图像，512维特征)
    image_features = model.encode_image(image_input)

    # 提取文本特征
    # model.encode_text(): 使用CLIP文本编码器提取特征
    # 输入形状: [100, 77] (100个文本，每个77个token)
    # 输出形状: [100, 512] (100个文本特征，每个512维)
    text_features = model.encode_text(text_inputs)

# 对特征进行L2归一化（单位化）
# .norm(dim=-1, keepdim=True): 计算每个特征向量的L2范数
# /= : 原地除法，使每个特征向量成为单位向量
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 计算相似度并转换为概率
# image_features @ text_features.T: 矩阵乘法计算相似度
#    [1, 512] × [512, 100] = [1, 100]
# 100.0 * : 温度缩放（相当于温度参数τ=0.01）
# softmax(dim=-1): 在最后一个维度（类别维度）应用softmax
# 得到图像属于每个类别的概率
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 获取前5个最相似的类别
# similarity[0]: 取第一个（也是唯一一个）图像的相似度向量
# topk(5): 获取前5个最大值和对应的索引
values, indices = similarity[0].topk(5)

# 打印结果
print("\nTop predictions:\n")
# 遍历前5个预测结果
for value, index in zip(values, indices):
    # cifar100.classes[index]: 获取类别名称
    # value.item(): 获取概率值（从张量转换为Python浮点数）
    # 100 * value.item(): 转换为百分比
    # :>16s: 格式化输出，右对齐，宽度16字符
    # :.2f%: 保留2位小数的百分比
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")