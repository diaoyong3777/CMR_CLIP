# 导入必要的测试库和模块
import numpy as np  # 数值计算，用于比较数组
import pytest  # Python测试框架
import torch  # PyTorch深度学习框架
from PIL import Image  # 图像处理库

import clip  # 导入要测试的CLIP模块


# 使用pytest的参数化装饰器：为每个可用模型运行一次测试
# model_name参数会依次取clip.available_models()返回的所有模型名
@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    """测试JIT模型和Python模型输出的一致性"""

    # 指定测试设备为CPU（确保测试稳定，不依赖GPU）
    device = "cpu"

    # 1. 加载JIT（Just-In-Time编译）版本的模型
    # jit=True: 加载优化过的序列化模型，执行效率高但不可修改
    jit_model, transform = clip.load(model_name, device=device, jit=True)

    # 2. 加载普通Python版本的模型
    # jit=False: 加载标准的PyTorch模型，可修改但执行效率较低
    py_model, _ = clip.load(model_name, device=device, jit=False)

    # 3. 准备测试数据
    # 加载项目中的示例图像"CLIP.png"
    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # unsqueeze(0): 添加批次维度 [C,H,W] → [1,C,H,W]
    # to(device): 将张量移动到指定设备（这里就是CPU）

    # 对三个文本进行分词
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # 三个文本描述，与CLIP.png图像内容相关

    # 4. 进行推理（不计算梯度，节省内存）
    with torch.no_grad():
        # 使用JIT模型进行推理
        logits_per_image, _ = jit_model(image, text)
        # 计算softmax得到概率分布
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # .cpu(): 确保张量在CPU上
        # .numpy(): 转换为NumPy数组便于比较

        # 使用Python模型进行推理
        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 5. 断言验证：检查两个模型的输出是否足够接近
    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)
    # np.allclose: 比较两个数组是否在容忍范围内相等
    # atol=0.01: 绝对容忍误差（absolute tolerance）0.01
    # rtol=0.1: 相对容忍误差（relative tolerance）10%

# 为什么要测试这个？
# CLIP提供了两种模型加载方式：
# 方式A: jit=True - 加载预编译的优化模型（.pt文件）
#        优点：运行速度快，部署友好
#        缺点：不可修改，调试困难

# 方式B: jit=False - 加载标准PyTorch模型（从state_dict构建）
#        优点：可修改，易于调试和理解
#        缺点：运行速度较慢

# 测试目标：确保两种方式得到相同的结果
# 如果结果不一致，说明：
# 1. JIT编译可能改变了计算逻辑
# 2. 模型权重加载有问题
# 3. 存在bug