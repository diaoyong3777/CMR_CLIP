# 导入PyTorch深度学习框架
import torch
# 从torch.utils.data中导入Dataset和DataLoader类，用于数据加载
from torch.utils.data import Dataset, DataLoader
# 导入random模块，用于生成随机数
import random

# 导入pickle模块，用于Python对象的序列化和反序列化操作
import pickle #
# 导入os模块，提供与操作系统交互的功能，如文件和目录操作
import os
# 导入PIL库中的Image模块，用于图像处理操作
from PIL import Image
# 图像预处理
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, ToTensor, Normalize, InterpolationMode
from typing import Union, List  # 类型注解
from clip.simple_tokenizer import SimpleTokenizer as Tokenizer # 需要用到CLIP的分词器和词汇表，本load文件放到clip目录外
# 创建分词器实例【整个文件就创建这一个】
_tokenizer = Tokenizer()

# CLIP的文本分词函数【文本=>ID序列】【truncate=True允许截断】
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True):
    # 如果输入是单个字符串，转换为列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊token的ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]  # 文本开始标记
    eot_token = _tokenizer.encoder["<|endoftext|>"]  # 文本结束标记

    # 对每个文本进行分词，并添加特殊标记【文本=>ID序列=>加了开始结束的ID序列】
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 填充结果张量
    for i, tokens in enumerate(all_tokens):
        # 检查是否超过上下文长度
        if len(tokens) > context_length:
            if truncate:
                # 截断并确保最后一个token是EOT
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                # 不截断则报错
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        # 将token序列复制到结果张量中
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result  # 返回分词结果【多个文本 × 77个序列】

# Dataset
class CMRDateset(Dataset):
    def __init__(self, data, images_dir, is_train=True, image_resolution=224, context_length=77,seed=0):
        self.images_dir = images_dir
        self.is_train = is_train
        # 加载数据
        self.images = data["indexs"]
        self.texts = data["captions"]
        self.labels = data["labels"]
        # 图像预处理
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            RandomCrop(image_resolution) if is_train else CenterCrop(image_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # 设置随机种子
        self.random_state = random.Random(seed) if seed is not None else random.Random()


    def __len__(self):
        return len(self.images)

    # 获取图片和路径
    def getImage(self, idx):
        image_path = os.path.join(self.images_dir,self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image,image_path
    # 获取文本
    def getText(self, idx):
        text = self.texts[idx]
        # 随机选择一个文本【COCO一张图片对应有5段描述】
        if isinstance(text, list) and len(text) > 0:
            text = self.random_state.choice(text)
        else:
            # 如果只有一个文本或者不是列表，直接使用
            text = text
        # tokenize分词返回列表，这里只有一个文本取元素即可
        text = tokenize(text)[0]
        return text
    # 获取标签
    def getLabel(self, idx):
        label = self.labels[idx]
        label = torch.from_numpy(label)
        return label
    # 一次性获取所有样本的标签
    def getAllLabel(self):
        labels = torch.zeros([self.__len__(), len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels


    def __getitem__(self, idx):
        image,image_path = self.getImage(idx)
        text = self.getText(idx)
        label = self.getLabel(idx)
        return  idx,image, text, label

# 随机划分数据集为查询集、训练集和检索集【一部分查询，剩余检索，训练从检索集中取一部分】
def split_data(data, query_num=5000, train_num=10000, seed=None):
    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 创建索引列表并打乱  # range(len(indexs))生成[0, 1, 2, ..., num_samples-1]
    indices = list(range(len(data["indexs"])))
    random.shuffle(indices)  # 原地打乱，返回None

    # 划分索引范围
    # 查询集：前query_num个
    query_index = indices[: query_num]
    # 检索集：从query_num开始到结束（包含训练集）
    retrieval_index = indices[query_num:]
    # 训练集：从query_num开始，取train_num个
    train_index = indices[query_num: query_num + train_num]
    # print(query_index[0:10])
    # 使用列表推导式提取子集
    data = {
        "query_data": {
            "indexs": [data["indexs"][i] for i in query_index],
            "captions": [data["captions"][i] for i in query_index],
            "labels": [data["labels"][i] for i in query_index]
        },
        "retrieval_data": {
            "indexs": [data["indexs"][i] for i in retrieval_index],
            "captions": [data["captions"][i] for i in retrieval_index],
            "labels": [data["labels"][i] for i in retrieval_index]
        },
        "train_data": {
            "indexs": [data["indexs"][i] for i in train_index],
            "captions": [data["captions"][i] for i in train_index],
            "labels": [data["labels"][i] for i in train_index]
        }
    }

    return data


# 加载pkl数据到CLIP
def CMRDataLoader(dataset,batch_size,is_train):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,  # 使用锁页内存，加速GPU传输
        shuffle=True if is_train else False  #  # 训练集，每个 epoch 开始时打乱数据顺序【训练时还是会有随机性】
    )
    return dataloader



# 当前脚本文件中做测试
if __name__ == '__main__':
    pkl_dataset = "./pkl_dataset/flickr25k.pkl"
    images_dir = "./raw_dataset/mirflickr25k"
    # pkl_dataset = "./pkl_dataset/coco2017.pkl"
    # images_dir = "C:\\Users\\dy\\Desktop\\CMR_BASE\\dataset\\raw_dataset\\coco2017"
    # pkl_dataset = "./pkl_dataset/nuswide.pkl"
    # images_dir = "C:\\Users\\dy\\Desktop\\CMR_BASE\\dataset\\raw_dataset\\nuswide"
    # 加载数据
    with open(pkl_dataset, "rb") as f:
        data = pickle.load(f)
    # 划分查询集、检索集、训练集
    data = split_data(data, seed=0)
    # 分别创建Dataset
    query_dataset = CMRDateset(data["query_data"], images_dir=images_dir, is_train=False)
    retrieval_dataset = CMRDateset(data["retrieval_data"], images_dir=images_dir, is_train=False)
    train_dataset = CMRDateset(data["train_data"], images_dir=images_dir, is_train=True)
    # 分别创建Dataloader
    batch_size = 32
    query_dataloader = CMRDataLoader(query_dataset,batch_size, is_train=False)
    retrieval_dataloader = CMRDataLoader(retrieval_dataset,batch_size, is_train=False)
    train_dataloader = CMRDataLoader(train_dataset,batch_size, is_train=True)


    for i,(idxs,images, texts, labels) in enumerate(query_dataloader):
        if i == 0:
            print(images.size(), texts.size(), labels.size())
            print(idxs[0])
            print(images[0])
            print(texts[0])
            print(query_dataset.getImage(0))
            print(query_dataset.getText(0))
            print(query_dataset.getLabel(0))
            print(query_dataset.getAllLabel().size(),query_dataset.getAllLabel()[0:10])
        if i == 1000:
            break

# D:\Anaconda3\envs\study\python.exe C:\Users\dy\Desktop\CLIP\load_data_to_clip.py
# [386, 3654, 4469, 19861, 15804, 17846, 3090, 15715, 12653, 8668]
# torch.Size([32, 3, 224, 224]) torch.Size([32, 1, 77]) torch.Size([32, 24])
# tensor(0)
# tensor([[[ 0.6895,  0.6895,  0.6895,  ..., -0.6536, -0.5806, -0.6098],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.6682, -0.6974, -0.6244],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.5952, -0.6828, -0.7412],
#          ...,
#          [ 0.6895,  0.6895,  0.6895,  ..., -1.0769, -0.9748, -0.6098],
#          [ 0.6895,  0.6895,  0.6895,  ..., -1.0039, -0.9456, -0.6244],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.7266, -1.0477, -0.7704]],
#
#         [[ 0.7242,  0.7242,  0.7242,  ..., -0.9117, -0.8516, -0.8967],
#          [ 0.7242,  0.7242,  0.7242,  ..., -0.9417, -0.9867, -0.8816],
#          [ 0.7242,  0.7242,  0.7242,  ..., -0.8516, -0.9867, -1.0317],
#          ...,
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.4069, -1.3169, -0.8816],
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.3319, -1.3019, -0.9117],
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.0167, -1.3919, -1.0767]],
#
#         [[ 0.8092,  0.8092,  0.8092,  ..., -0.4990, -0.4422, -0.4706],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.5133, -0.5559, -0.4706],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.4422, -0.5417, -0.5986],
#          ...,
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.8261, -0.7266, -0.4564],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.7834, -0.7408, -0.4848],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.5417, -0.8545, -0.5986]]])
# tensor([[49406, 49407,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]], dtype=torch.int32)
# (tensor([[[ 0.6895,  0.6895,  0.6895,  ..., -0.6536, -0.5806, -0.6098],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.6682, -0.6974, -0.6244],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.5952, -0.6828, -0.7412],
#          ...,
#          [ 0.6895,  0.6895,  0.6895,  ..., -1.0769, -0.9748, -0.6098],
#          [ 0.6895,  0.6895,  0.6895,  ..., -1.0039, -0.9456, -0.6244],
#          [ 0.6895,  0.6895,  0.6895,  ..., -0.7266, -1.0477, -0.7704]],
#
#         [[ 0.7242,  0.7242,  0.7242,  ..., -0.9117, -0.8516, -0.8967],
#          [ 0.7242,  0.7242,  0.7242,  ..., -0.9417, -0.9867, -0.8816],
#          [ 0.7242,  0.7242,  0.7242,  ..., -0.8516, -0.9867, -1.0317],
#          ...,
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.4069, -1.3169, -0.8816],
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.3319, -1.3019, -0.9117],
#          [ 0.7242,  0.7242,  0.7242,  ..., -1.0167, -1.3919, -1.0767]],
#
#         [[ 0.8092,  0.8092,  0.8092,  ..., -0.4990, -0.4422, -0.4706],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.5133, -0.5559, -0.4706],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.4422, -0.5417, -0.5986],
#          ...,
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.8261, -0.7266, -0.4564],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.7834, -0.7408, -0.4848],
#          [ 0.8092,  0.8092,  0.8092,  ..., -0.5417, -0.8545, -0.5986]]]), './raw_dataset/mirflickr25k\\mirflickr/im397.jpg')
# tensor([[49406, 49407,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]], dtype=torch.int32)
# tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#        dtype=torch.int8)
# torch.Size([5000, 24]) tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]])
#
# 进程已结束，退出代码为 0
