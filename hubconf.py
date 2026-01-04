# 导入CLIP模块中的核心函数
# _tokenize: 文本分词器，将文本转换为token
# _load: 模型加载函数，负责创建模型并加载预训练权重
# _available_models: 返回所有可用模型名称的函数
from clip.clip import tokenize as _tokenize, load as _load, available_models as _available_models

# 导入正则表达式模块，用于处理模型名称中的特殊字符
import re

# 导入字符串模块，用于获取标点符号的定义
import string

# 定义依赖项：使用这个hub文件需要安装的Python包
# torch: PyTorch深度学习框架
# torchvision: PyTorch的视觉处理库（包含图像预处理函数）
# ftfy: 修复乱码文本的工具（Fix Text For You）
# regex: 增强版的正则表达式库
# tqdm: 进度条显示库
dependencies = ["torch", "torchvision", "ftfy", "regex", "tqdm"]

# 为所有可用模型创建兼容的函数名
# 由于函数名不能包含特殊字符（如斜杠/），需要将模型名中的特殊字符替换为下划线
# 例如：模型名"ViT-B/32" -> 函数名"ViT_B_32"
model_functions = {
    model: re.sub(f'[{string.punctuation}]', '_', model)  # 将所有标点符号替换为下划线
    for model in _available_models()  # 遍历所有可用的CLIP模型
}


# 创建一个模型加载函数的工厂函数
# 这个函数会根据指定的模型名生成对应的加载函数
def _create_hub_entrypoint(model):
    # 定义实际的模型加载函数（闭包函数）
    def entrypoint(**kwargs):
        # 调用CLIP模块内部的_load函数加载指定模型
        # model参数：要加载的模型名称（如"ViT-B/32"）
        # kwargs：传递给_load的其他参数（如device, jit, download_root等）
        return _load(model, **kwargs)

    # 为生成的函数添加文档字符串（help文档）
    # 这样用户使用help()或查看文档时能看到说明
    entrypoint.__doc__ = f"""Loads the {model} CLIP model

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to put the loaded model (模型加载的设备，如"cuda"或"cpu")

        jit : bool
            Whether to load the optimized JIT model or more hackable non-JIT model (default).
            (是否加载JIT优化版本，JIT版本更快但不可修改，非JIT版本更灵活)

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"
            (模型权重下载路径，默认使用用户缓存目录)

        Returns
        -------
        model : torch.nn.Module
            The {model} CLIP model (CLIP模型实例)

        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
            (图像预处理函数，将PIL图像转换为模型可接受的张量)
        """
    return entrypoint  # 返回生成的加载函数


# 定义一个tokenize函数，直接返回CLIP模块中的_tokenize函数
# 这样用户可以通过torch.hub.load(...)获取分词器
def tokenize():
    return _tokenize


# 为所有模型创建入口点函数的字典
# key: 兼容的函数名（如"ViT_B_32"）
# value: 对应的模型加载函数
_entrypoints = {
    model_functions[model]: _create_hub_entrypoint(model)  # 为每个模型创建加载函数
    for model in _available_models()  # 遍历所有模型
}

# 将创建的入口点函数添加到全局命名空间
# 这样PyTorch Hub就能找到这些函数并调用它们
# 例如：当用户调用torch.hub.load('openai/CLIP', 'ViT_B_32')时
# PyTorch Hub会在globals()中查找名为'ViT_B_32'的函数
globals().update(_entrypoints)


# # 用户代码示例：
# import torch
#
# # 情况1：加载ViT-B/32模型（论文中的Vision Transformer Base）
# # 实际调用流程：
# # 1. torch.hub查找openai/CLIP仓库
# # 2. 在hubconf.py中查找名为"ViT_B_32"的函数
# # 3. 调用_create_hub_entrypoint("ViT-B/32")创建的闭包函数
# # 4. 闭包函数内部调用_load("ViT-B/32", device="cuda", ...)
# model, preprocess = torch.hub.load('openai/CLIP', 'ViT_B_32', device='cuda')
#
# # 情况2：加载ResNet50模型
# # 函数名映射：RN50 -> RN50（没有特殊字符，保持不变）
# model, preprocess = torch.hub.load('openai/CLIP', 'RN50')
#
# # 情况3：获取分词器
# # 直接调用tokenize()函数
# tokenizer = torch.hub.load('openai/CLIP', 'tokenize')