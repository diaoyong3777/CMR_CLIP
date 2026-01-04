# 导入PyTorch深度学习框架
import torch
# 导入OpenAI的CLIP库，用于多模态（图像-文本）学习
# 查找clip模块的位置
import clip
print(f"\nclip模块位置: {clip.__file__}")
# 导入PIL库的Image模块，用于图像处理
from PIL import Image

# 检测并设置计算设备
# torch.cuda.is_available(): 检查CUDA（GPU）是否可用
# 如果可用，device="cuda"；否则device="cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型和预处理函数
# clip.load("ViT-B/32", device=device):
#   "ViT-B/32": Vision Transformer Base模型，patch大小为32x32
#   device=device: 指定模型加载到的设备（GPU或CPU）
# 返回:
#   model: 加载的CLIP模型
#   preprocess: 图像预处理函数（调整大小、裁剪、归一化等）
model, preprocess = clip.load("ViT-B/32", device=device)

# 预处理图像并准备输入
# 步骤分解:
# 1. Image.open("CLIP.png"): 打开名为"CLIP.png"的图像文件，返回PIL Image对象
# 2. preprocess(...): 应用CLIP预处理（调整到224x224、中心裁剪、转换为张量、归一化）
# 3. .unsqueeze(0): 在第0维度添加批次维度
#    从形状[3, 224, 224]变为[1, 3, 224, 224]
#    因为模型需要批次输入，即使只有一张图像
# 4. .to(device): 将图像张量移动到指定设备（GPU或CPU）
image = preprocess(Image.open("../CLIP.png")).unsqueeze(0).to(device)

# 准备文本输入
# clip.tokenize(["a diagram", "a dog", "a cat"]):
#   将文本列表转换为token ID张量
#   每个文本被分词并填充到长度77（CLIP标准）
#   输出形状: [3, 77]（3个文本，每个77个token）
# .to(device): 将文本张量移动到指定设备
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 禁用梯度计算，进入推理模式
with torch.no_grad():
    # 提取图像特征
    # model.encode_image(image): 使用CLIP图像编码器提取特征
    # 输入: [1, 3, 224, 224]  # 1张图像，3通道，224x224
    # 输出: [1, 512]  # 1个512维特征向量
    image_features = model.encode_image(image)

    # 提取文本特征
    # model.encode_text(text): 使用CLIP文本编码器提取特征
    # 输入: [3, 77]  # 3个文本，每个77个token
    # 输出: [3, 512]  # 3个512维特征向量
    text_features = model.encode_text(text)

    # 使用完整的CLIP模型计算图像-文本相似度
    # model(image, text): CLIP模型的正向传播
    #   输入图像和文本，返回相似度矩阵
    # 返回两个值:
    #   logits_per_image: 形状[1, 3]，图像与每个文本的相似度
    #   logits_per_text: 形状[3, 1]，每个文本与图像的相似度
    # 这两个矩阵互为转置，只是维度顺序不同
    logits_per_image, logits_per_text = model(image, text)

    # 将相似度转换为概率分布
    # logits_per_image.softmax(dim=-1):
    #   在最后一个维度（文本维度）上应用softmax
    #   将相似度分数转换为概率，每行和为1
    # .cpu().numpy(): 将结果从GPU移动到CPU，并转换为NumPy数组
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印分类概率
# probs形状: [1, 3]，表示一张图像属于3个文本描述的概率
print("Label probs:", probs)  # 输出: [[0.9927937  0.00421068 0.00299572]]