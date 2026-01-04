# 导入必要的模块
import hashlib  # 用于计算文件的SHA256哈希值，验证下载完整性
import os  # 操作系统接口，用于文件路径操作
import urllib  # 用于下载文件的库
import warnings  # 警告信息处理
from packaging import version  # 版本比较工具
from typing import Union, List  # 类型注解

import torch  # PyTorch深度学习框架
from PIL import Image  # Python图像处理库
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  # 图像预处理
from tqdm import tqdm  # 进度条显示库

# 导入当前包内的模块
from .model import build_model  # 从model模块导入模型构建函数
from .simple_tokenizer import SimpleTokenizer as _Tokenizer  # 导入分词器并重命名

# 尝试导入torchvision的InterpolationMode枚举，用于指定图像插值方式【Resize放缩算法】
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC  # 双三次插值，图像质量较好
except ImportError:
    # 如果torchvision版本较旧，使用PIL的BICUBIC常量
    BICUBIC = Image.BICUBIC

# 检查PyTorch版本，如果低于1.7.1则发出警告
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

# 定义模块的公共接口：哪些函数/变量可以被外部访问
__all__ = ["available_models", "load", "tokenize"]

# 创建分词器实例（单例模式）
_tokenizer = _Tokenizer()

# CLIP预训练模型的下载链接字典【ViT-B-32.pt这个不只是视觉模型，是使用这个ViT训练得到的整个模型文件】
# 键：模型名称，值：模型权重文件的下载URL
_MODELS = {
    # ResNet系列模型
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    # Vision Transformer系列模型
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


# 下载函数：从URL下载文件到本地目录
def _download(url: str, root: str):
    """下载模型文件到指定目录"""
    # 创建目录（如果不存在）
    os.makedirs(root, exist_ok=True)
    # 从URL提取文件名（最后一部分）
    filename = os.path.basename(url)

    # 从URL中提取预期的SHA256哈希值（URL的倒数第二部分）
    expected_sha256 = url.split("/")[-2]
    # 完整的下载目标路径
    download_target = os.path.join(root, filename)

    # 检查目标路径是否存在且不是普通文件（可能是目录）【为了避免xxx.pt是一个目录】
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 如果文件已存在，验证其SHA256哈希值
    if os.path.isfile(download_target):
        # 计算现有文件的SHA256哈希值
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target  # 哈希值匹配，直接返回文件路径
        else:
            # 哈希值不匹配，发出警告并重新下载
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 下载文件：使用urllib打开URL连接，写入本地文件
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 创建进度条，显示下载进度
        # total: 总大小（从HTTP头获取）
        # ncols: 进度条宽度
        # unit='iB': 单位为字节
        # unit_scale=True: 自动缩放单位（KB, MB等）
        # unit_divisor=1024: 使用1024作为缩放基数
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)  # 每次读取8KB数据
                if not buffer:  # 如果没有数据了，结束循环
                    break

                output.write(buffer)  # 写入文件
                loop.update(len(buffer))  # 更新进度条

    # 下载完成后验证文件的SHA256哈希值
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target  # 返回下载的文件路径


# 图像转换函数：将图像转换为RGB模式
def _convert_image_to_rgb(image):
    """确保图像是RGB模式，去除Alpha通道等"""
    return image.convert("RGB")


# 图像预处理转换函数：根据输入尺寸创建预处理流水线
def _transform(n_px):
    """创建图像预处理转换流水线"""
    return Compose([
        # 1. 调整图像大小到n_pxn_px，使用双三次插值【放缩算法技术】
        Resize(n_px, interpolation=BICUBIC),
        # 2. 中心裁剪到n_pxn_px
        CenterCrop(n_px),
        # 3. 确保图像是RGB模式
        _convert_image_to_rgb,
        # 4. 将PIL图像转换为PyTorch张量（形状从HWC变为CHW，值从0-255变为0-1）
        ToTensor(),
        # 5. 标准化：使用CLIP训练时的均值和标准差【# 1. 加速收敛：使数据分布更稳定 # 2. 数值稳定：防止梯度爆炸/消失】
        # 第一个元组是RGB通道的均值
        # 第二个元组是RGB通道的标准差
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # 来源：在CLIP训练数据集上计算得到
        # 1. 收集所有训练图像
        # 2. 对每个图像计算RGB通道的均值和标准差
        # 3. 对所有图像取平均
    ])


# 获取可用模型列表
def available_models() -> List[str]:
    """返回可用的CLIP模型名称列表"""
    return list(_MODELS.keys())  # 返回_MODELS字典的所有键


# 主函数：加载CLIP模型
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None):
    """
    加载CLIP模型

    参数
    ----------
    name : str
        视觉端模型名称（通过clip.available_models()获取），或包含state_dict的模型检查点路径

    device : Union[str, torch.device]
        加载模型的目标设备

    jit : bool
        是否加载优化的JIT模型或更易修改的非JIT模型（默认）

    download_root: str
        模型文件下载路径；默认使用"~/.cache/clip"

    返回
    -------
    model : torch.nn.Module
        CLIP模型

    preprocess : Callable[[PIL.Image], torch.Tensor]
        图像预处理函数，将PIL图像转换为模型可接受的张量
    """
    # 确定模型文件路径
    if name in _MODELS:
        # 如果是列表中的预定义模型，下载或使用缓存
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 如果name是文件路径，直接使用
        model_path = name
    else:
        # 既不是预定义模型也不是有效文件路径，报错
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 加载CLIP模型文件
    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试作为JIT（Just-In-Time）编译的模型加载
            # JIT模型是经过优化的序列化模型，不可修改但执行效率高
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None  # JIT模型没有单独的state_dict
        except RuntimeError:
            # 如果不是JIT模型，尝试作为状态字典加载
            if jit:
                # 用户要求JIT但文件不是JIT格式，发出警告
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")  # 加载到CPU内存

    # 如果不需要JIT，构建可修改的CLIP模型
    if not jit:
        # 使用build_model函数构建模型并加载权重
        model = build_model(state_dict or model.state_dict()).to(device)

        # 如果在CPU上运行，确保模型是float32（而非默认的float16）
        if str(device) == "cpu":
            model.float()

        # 返回模型和对应的图像预处理函数
        # model.visual.input_resolution: 视觉编码器期望的输入图像分辨率
        return model, _transform(model.visual.input_resolution)

    # 以下是JIT模型的特殊处理代码
    # 当加载JIT模型时，需要修补设备信息

    # 创建设备占位符，用于获取设备相关的计算图节点
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    # 找到设备节点
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    # 辅助函数：获取计算图节点的属性
    def _node_get(node: torch._C.Node, key: str):
        """获取节点的属性，处理不同类型的返回值"""
        sel = node.kindOf(key)  # 获取属性的类型
        return getattr(node, sel)(key)  # 动态调用对应的方法获取属性值

    # 修补函数：将计算图中的设备信息替换为指定的设备
    def patch_device(module):
        try:
            # 获取模块的计算图
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        # 某些JIT模型可能有多个计算图
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        # 遍历所有计算图
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                # 查找设备常量节点（值以"cuda"开头）
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)  # 替换设备属性

    # 应用设备修补到整个模型
    model.apply(patch_device)
    # 单独修补编码函数
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 如果在CPU上运行，还需要修补数据类型
    if str(device) == "cpu":
        # 创建float32占位符，用于获取float32相关的计算图节点
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        # 找到float32节点
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        # 修补函数：将计算图中的数据类型替换为float32
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):  # 查找类型转换节点
                    inputs = list(node.inputs())
                    # dtype参数可能是第二个或第三个输入
                    for i in [1, 2]:
                        # 5对应torch.float16，需要替换为float32
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)  # 替换为float32

        # 应用数据类型修补
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # 确保模型权重是float32
        model.float()

    # 返回JIT模型和图像预处理函数
    # model.input_resolution.item(): 从张量中获取分辨率值
    return model, _transform(model.input_resolution.item())


# 文本分词函数【文本=>ID序列】
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    返回给定输入字符串的分词表示

    参数
    ----------
    texts : Union[str, List[str]]
        要分词的输入字符串或字符串列表

    context_length : int
        上下文长度；所有CLIP模型使用77作为上下文长度

    truncate: bool
        如果编码长度超过上下文长度，是否截断文本

    返回
    -------
    包含结果token的二维张量，形状 = [输入字符串数量, context_length]
    当torch版本<1.8.0时返回LongTensor，因为旧版本的index_select需要long类型的索引
    """
    # 如果输入是单个字符串，转换为列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊token的ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]  # 文本开始标记
    eot_token = _tokenizer.encoder["<|endoftext|>"]  # 文本结束标记

    # 对每个文本进行分词，并添加特殊标记【文本=>ID序列=>加了开始结束的ID序列】
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # 根据PyTorch版本选择合适的整数类型
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        # 旧版本需要torch.long
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        # 新版本可以使用torch.int
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