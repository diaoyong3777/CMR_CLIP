# 导入必要的模块
from collections import OrderedDict  # 有序字典，保持层定义的顺序
from typing import Tuple, Union  # 类型注解，用于函数参数类型提示

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数接口
from torch import nn  # PyTorch神经网络模块


# 定义瓶颈块（Bottleneck Block），用于ResNet
# 这是标准的ResNet残差块结构
class Bottleneck(nn.Module):
    expansion = 4  # 扩展因子：输出通道数是输入通道数的4倍

    # 初始化瓶颈块
    # inplanes: 输入通道数
    # planes: 中间层通道数（实际输出通道数 = planes * expansion）
    # stride: 步长，用于下采样
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()  # 调用父类nn.Module的初始化

        # 所有卷积层的步长都是1，当stride>1时，在第二个卷积后进行平均池化
        # 这是为了消除混叠效应（anti-aliasing）

        # 第一个卷积：1x1卷积，用于降维
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        self.relu1 = nn.ReLU(inplace=True)  # ReLU激活，inplace=True节省内存

        # 第二个卷积：3x3卷积，用于特征提取
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 平均池化层：当stride>1时用于下采样
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个卷积：1x1卷积，用于升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        # 下采样路径：当输入输出维度不匹配时需要
        self.downsample = None
        self.stride = stride

        # 需要下采样的情况：stride>1（空间尺寸变化）或通道数不匹配
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # 下采样层：先平均池化，然后1x1卷积（stride=1）
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # 平均池化下采样
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    # 前向传播函数
    # x: 输入张量，形状为 [batch_size, in_channels, height, width]
    def forward(self, x: torch.Tensor):
        identity = x  # 保存输入用于残差连接

        # 主路径：三个卷积层
        out = self.relu1(self.bn1(self.conv1(x)))  # 1x1卷积 -> BN -> ReLU
        out = self.relu2(self.bn2(self.conv2(out)))  # 3x3卷积 -> BN -> ReLU
        out = self.avgpool(out)  # 如果需要则下采样
        out = self.bn3(self.conv3(out))  # 1x1卷积 -> BN

        # 如果存在下采样路径，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：主路径输出 + 下采样后的输入
        out += identity
        out = self.relu3(out)  # 最后的ReLU激活

        return out  # 输出形状：[batch_size, planes*4, height/stride, width/stride]



# 注意力池化层
# 使用注意力机制代替传统的全局平均池化
class AttentionPool2d(nn.Module):
    # 初始化注意力池化层
    # spacial_dim: 空间维度大小（输入特征图的大小）
    # embed_dim: 嵌入维度（特征维度）
    # num_heads: 注意力头数
    # output_dim: 输出维度（默认与embed_dim相同）
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 位置编码：为每个空间位置和CLS token学习位置信息
        # 形状: [spacial_dim**2 + 1, embed_dim]
        # 除以sqrt(embed_dim)进行缩放初始化
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        # 注意力机制的投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # Key投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Query投影
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Value投影
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)  # 输出投影
        self.num_heads = num_heads  # 注意力头数

    # 前向传播
    # x: 输入特征图，形状为 [batch_size, channels, height, width]
    def forward(self, x):
        # 将特征图从NCHW格式转换为序列格式
        # flatten(start_dim=2): 将空间维度展平，形状变为 [N, C, H*W]
        # permute(2, 0, 1): 重新排列为 [(H*W), N, C]
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC

        # 添加CLS token：在序列开头添加全局平均池化特征
        # mean(dim=0): 计算批次中所有样本的平均，形状 [1, N, C]
        # cat: 拼接，形状变为 [(H*W+1), N, C]
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        # 添加位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        # 使用多头注意力机制
        # query取CLS token，key和value取所有token
        # CLS token作为查询，收集整个特征图的信息
        x, _ = F.multi_head_attention_forward(
            query=x[:1],  # 只使用CLS token作为查询 [1, N, C]
            key=x,  # 所有token作为key [(HW+1), N, C]
            value=x,  # 所有token作为value [(HW+1), N, C]
            embed_dim_to_check=x.shape[-1],  # 检查特征维度
            num_heads=self.num_heads,  # 注意力头数
            q_proj_weight=self.q_proj.weight,  # Q投影权重
            k_proj_weight=self.k_proj.weight,  # K投影权重
            v_proj_weight=self.v_proj.weight,  # V投影权重
            in_proj_weight=None,  # 不使用合并的投影
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),  # 投影偏置
            bias_k=None,  # 无key偏置
            bias_v=None,  # 无value偏置
            add_zero_attn=False,  # 不添加零注意力
            dropout_p=0,  # 无dropout
            out_proj_weight=self.c_proj.weight,  # 输出投影权重
            out_proj_bias=self.c_proj.bias,  # 输出投影偏置
            use_separate_proj_weight=True,  # 使用分离的投影权重
            training=self.training,  # 训练模式标志
            need_weights=False  # 不需要返回注意力权重
        )

        # 去掉序列维度，返回CLS token的特征
        return x.squeeze(0)  # 形状: [N, C]


# 改进的ResNet类【用到前面两个类】
class ModifiedResNet(nn.Module):
    """
    与torchvision的ResNet类似但有以下改进：
    - 使用3个"stem"卷积而不是1个，使用平均池化而不是最大池化
    - 执行抗混叠步长卷积，当stride>1时在卷积前添加平均池化
    - 最终的池化层是QKV注意力而不是平均池化
    """

    # 初始化ModifiedResNet
    # layers: 每个阶段的块数，如[3, 4, 6, 3]表示ResNet50
    # output_dim: 输出特征维度
    # heads: 注意力头数
    # input_resolution: 输入图像分辨率（默认224）
    # width: 基础宽度（第一个卷积后的通道数）
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 3层stem卷积（代替传统的1层卷积+池化）
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)  # 3->32通道，stride=2
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)  # 32->32通道
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)  # 32->64通道
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(2)  # 平均池化，stride=2

        # 残差层
        self._inplanes = width  # 当前输入通道数（可变，在构建层时会更新）

        # 构建4个残差阶段
        self.layer1 = self._make_layer(width, layers[0])  # 第一阶段：不降采样
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # 第二阶段：降采样
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # 第三阶段：降采样
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # 第四阶段：降采样

        # 注意力池化层
        embed_dim = width * 32  # ResNet特征维度（64*32=2048）
        # input_resolution // 32: 经过5次stride=2的下采样，224->7
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    # 构建一个残差阶段
    # planes: 该阶段的基础通道数
    # blocks: 该阶段的块数
    # stride: 第一个块的步长（用于下采样）
    def _make_layer(self, planes, blocks, stride=1):
        # 第一个块使用指定的stride（可能下采样）
        layers = [Bottleneck(self._inplanes, planes, stride)]

        # 更新当前输入通道数
        self._inplanes = planes * Bottleneck.expansion

        # 添加剩余的块（stride=1，不下采样）
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        # 将层组合为Sequential模块
        return nn.Sequential(*layers)

    # 前向传播
    # x: 输入图像，形状 [N, 3, H, W]
    def forward(self, x):
        # stem函数：处理3层stem卷积
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))  # 第一层
            x = self.relu2(self.bn2(self.conv2(x)))  # 第二层
            x = self.relu3(self.bn3(self.conv3(x)))  # 第三层
            x = self.avgpool(x)  # 池化
            return x  # 输出形状: [N, 64, H/4, W/4]

        # 确保输入数据类型与模型权重一致
        x = x.type(self.conv1.weight.dtype)

        # 通过整个网络
        x = stem(x)  # stem部分
        x = self.layer1(x)  # 第一阶段
        x = self.layer2(x)  # 第二阶段
        x = self.layer3(x)  # 第三阶段
        x = self.layer4(x)  # 第四阶段
        x = self.attnpool(x)  # 注意力池化

        return x  # 输出形状: [N, output_dim]


# 自定义LayerNorm，支持fp16
class LayerNorm(nn.LayerNorm):
    """继承torch的LayerNorm以处理fp16精度。"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  # 保存原始数据类型
        ret = super().forward(x.type(torch.float32))  # 用fp32计算LayerNorm
        return ret.type(orig_type)  # 转回原始数据类型


# 快速GELU激活函数
# 近似GELU但计算更快
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)  # sigmoid近似GELU


# 残差注意力块（Transformer块）
class ResidualAttentionBlock(nn.Module):
    # 初始化残差注意力块
    # d_model: 模型维度（特征维度）
    # n_head: 注意力头数
    # attn_mask: 注意力掩码（用于因果注意力）
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # 多头注意力层
        self.attn = nn.MultiheadAttention(d_model, n_head)

        # 第一个层归一化（注意力子层前）
        self.ln_1 = LayerNorm(d_model)

        # MLP（前馈网络）
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 扩展维度
            ("gelu", QuickGELU()),  # 激活函数
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 投影回原始维度
        ]))

        # 第二个层归一化（MLP子层前）
        self.ln_2 = LayerNorm(d_model)

        # 注意力掩码
        self.attn_mask = attn_mask

    # 注意力计算
    def attention(self, x: torch.Tensor):
        # 将注意力掩码移动到正确的设备和数据类型
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 自注意力计算：query、key、value都是x
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # 前向传播
    def forward(self, x: torch.Tensor):
        # 残差连接1：注意力子层
        x = x + self.attention(self.ln_1(x))
        # 残差连接2：MLP子层
        x = x + self.mlp(self.ln_2(x))
        return x


# Transformer编码器【=多个Transformer块】
class Transformer(nn.Module):
    # 初始化Transformer
    # width: 模型维度
    # layers: Transformer层数
    # heads: 注意力头数
    # attn_mask: 注意力掩码
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width  # 模型维度
        self.layers = layers  # 层数
        # 创建多个残差注意力块
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    # 前向传播
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# Vision Transformer (ViT)【可以发现ViT中的Transformer没有使用到 attn_mask: 注意力掩码】
class VisionTransformer(nn.Module):
    # 初始化Vision Transformer
    # input_resolution: 输入图像分辨率
    # patch_size: 图像块大小

    # width: 模型维度
    # layers: Transformer层数
    # heads: 注意力头数

    # output_dim: 输出维度
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        # 卷积层：将图像分割为patch并投影到特征空间【利用卷积层来巧妙实现图像分块，一个通道对应特征向量的一维】
        # kernel_size=patch_size, stride=patch_size: 实现patch分割
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 缩放因子用于初始化
        scale = width ** -0.5

        # 类别嵌入（CLS token）
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        # 位置编码：为每个patch和CLS token学习位置信息
        # (input_resolution // patch_size) ** 2: patch数量
        # +1: CLS token
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        # 层归一化（Transformer前）
        self.ln_pre = LayerNorm(width)

        # Transformer编码器
        self.transformer = Transformer(width, layers, heads)

        # 层归一化（Transformer后）
        self.ln_post = LayerNorm(width)

        # 投影矩阵：将ViT特征投影到共享嵌入空间【投影等价于无偏置的Linear层】
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    # 前向传播
    def forward(self, x: torch.Tensor):
        # 将图像分割为patch并投影
        x = self.conv1(x)  # 形状: [N, width, grid, grid]
        # 展平空间维度【批次-向量维数-图像块数】
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 形状: [N, width, grid**2]
        # 调整维度顺序【批次-图像块数-向量维数，一个样本就对应一个向量序列，可以送入Transformer中】
        x = x.permute(0, 2, 1)  # 形状: [N, grid**2, width]

        # 添加CLS token
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                       x], dim=1)  # 形状: [N, grid**2+1, width]【图像块数+1个向量】

        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)

        # 层归一化
        x = self.ln_pre(x)

        # Transformer编码器需要序列在第一个维度
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 取CLS token的特征【最终的提取到的特征向量】
        x = self.ln_post(x[:, 0, :])  # 形状: [N, width]

        # 投影到共享嵌入空间
        if self.proj is not None:
            x = x @ self.proj  # 形状: [N, output_dim]

        return x


# CLIP主模型
class CLIP(nn.Module):
    # 初始化CLIP模型
    # embed_dim: 嵌入维度（图像和文本特征的共同维度）

    # image_resolution: 图像分辨率
    # vision_layers: 视觉编码器层数（int或tuple）
    # vision_width: 视觉编码器宽度
    # vision_patch_size: ViT的patch大小（如果是ViT）

    # context_length: 文本上下文长度（最大token数）
    # vocab_size: 词汇表大小
    # transformer_width: 文本编码器宽度
    # transformer_heads: 文本编码器注意力头数
    # transformer_layers: 文本编码器层数
    def __init__(self,
                 embed_dim: int,
                 # 视觉编码器参数
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # 文本编码器参数
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()

        self.context_length = context_length  # 文本最大长度

        # 根据vision_layers的类型选择视觉编码器【元组/列表选用Resnet，一个整数选用ViT】
        if isinstance(vision_layers, (tuple, list)):
            # ResNet架构：vision_layers是元组，如[3, 4, 6, 3]
            vision_heads = vision_width * 32 // 64  # 计算注意力头数
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            # ViT架构：vision_layers是整数
            vision_heads = vision_width // 64  # 计算注意力头数
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 文本编码器（Transformer）
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 因果注意力掩码
        )

        # 文本相关参数
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # token嵌入
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置编码
        self.ln_final = LayerNorm(transformer_width)  # 最终层归一化

        # 投影参数
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 文本投影矩阵
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 温度参数，控制相似度范围

        # 初始化参数
        self.initialize_parameters()

    # 参数初始化
    def initialize_parameters(self):
        # 初始化token嵌入
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 初始化位置编码
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 如果使用ModifiedResNet，初始化其参数
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                # 初始化注意力池化层参数
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # 初始化ResNet残差块的BN参数
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):  # 最后一个BN层的权重
                        nn.init.zeros_(param)  # 初始化为0

        # 初始化Transformer参数
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            # 初始化注意力层参数
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            # 初始化MLP参数
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化文本投影矩阵
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # 构建注意力掩码（因果掩码）
    def build_attention_mask(self):
        # 惰性创建因果注意力掩码，视觉token之间完全可见
        # PyTorch使用加性注意力掩码；用-inf填充
        mask = torch.empty(self.context_length, self.context_length)  # 创建方阵
        mask.fill_(float("-inf"))  # 全部填充为负无穷
        # 保留上三角（对角线以上），下三角置为0
        mask.triu_(1)  # 参数1表示从主对角线上面1条对角线开始保留
        # tensor([[  0., -inf, -inf, -inf, -inf],
        #         [  0.,   0., -inf, -inf, -inf],
        #         [  0.,   0.,   0., -inf, -inf],
        #         [  0.,   0.,   0.,   0., -inf],
        #         [  0.,   0.,   0.,   0.,   0.]])
        return mask  # 形状: [context_length, context_length]

    # 模型的数据类型属性
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype  # 返回视觉编码器第一个卷积的权重数据类型

    # 编码图像：提取图像特征【视觉端输入：一批图像tensor】
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))  # 确保输入与模型权重类型一致

    # 编码文本：提取文本特征【文本端输入：一批文本ID序列】
    def encode_text(self, text):
        # token嵌入
        x = self.token_embedding(text).type(self.dtype)  # 形状: [batch_size, n_ctx, d_model]
        # nn.Embedding是一个查找表（lookup table）
        # 输入：整数索引（token IDs）
        # 输出：对应的嵌入向量【# 从权重矩阵中选取对应的行】

        # 添加位置编码
        x = x + self.positional_embedding.type(self.dtype)

        # Transformer需要序列在第一个维度
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 层归一化
        x = self.ln_final(x).type(self.dtype)

        # x的形状: [batch_size, n_ctx, transformer.width]
        # 取每个序列中EOT（end of text）标记的特征
        # text.argmax(dim=-1): 找到每个序列中最大的token ID（EOT token）【花式索引，找到每个样本的EOT】
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x  # 形状: [batch_size, embed_dim]

    # 前向传播：计算图像-文本相似度【模型返回批次大小的相似度矩阵】
    def forward(self, image, text):
        # 提取图像特征
        image_features = self.encode_image(image)
        # 提取文本特征
        text_features = self.encode_text(text)

        # 特征归一化（L2归一化）
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度作为logits
        logit_scale = self.logit_scale.exp()  # 温度参数取指数
        # 图像到文本的相似度【每个向量对应做点积】
        logits_per_image = logit_scale * image_features @ text_features.t()
        # 文本到图像的相似度（转置）
        logits_per_text = logits_per_image.t()

        # 返回两个相似度矩阵
        # logits_per_image形状: [batch_size, batch_size]
        # logits_per_text形状: [batch_size, batch_size]
        return logits_per_image, logits_per_text


# 将模型参数转换为fp16【可选。节省内存，加速计算】
def convert_weights(model: nn.Module):
    """将适用的模型参数转换为fp16精度"""

    def _convert_weights_to_fp16(l):
        # 转换卷积层和线性层的权重和偏置
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()  # 权重转为半精度
            if l.bias is not None:
                l.bias.data = l.bias.data.half()  # 偏置转为半精度

        # 转换多头注意力层的参数
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)  # 获取属性
                if tensor is not None:
                    tensor.data = tensor.data.half()  # 转为半精度

        # 转换投影矩阵
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):  # 检查是否有该属性
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()  # 转为半精度

    # 对模型的所有子模块应用转换函数
    model.apply(_convert_weights_to_fp16)


# 根据状态字典构建模型【加载CLIP预训练模型=ViT/ResNet+Transformer】
def build_model(state_dict: dict):
    # 检查是否为ViT架构（通过是否有"visual.proj"键判断）
    vit = "visual.proj" in state_dict

    # 根据架构类型解析模型参数
    if vit:
        # ViT架构
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # 视觉编码器宽度
        # 计算视觉编码器层数（通过注意力层数量）
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # patch大小
        # 计算grid大小
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size  # 图像分辨率
    else:
        # ResNet架构
        # 计算每个阶段的块数
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)  # 如(3, 4, 6, 3)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 视觉编码器宽度
        # 计算输出宽度
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None  # ResNet不使用patch
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]  # 验证
        image_resolution = output_width * 32  # 图像分辨率

    # 解析文本编码器参数
    embed_dim = state_dict["text_projection"].shape[1]  # 嵌入维度
    context_length = state_dict["positional_embedding"].shape[0]  # 上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 词汇表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 文本编码器宽度
    transformer_heads = transformer_width // 64  # 计算注意力头数
    # 计算文本编码器层数
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 创建CLIP模型实例
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 从状态字典中删除不需要的键，用于后续加载模型参数。这些不是模型参数，而是配置信息
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 转换权重为fp16
    convert_weights(model)
    # 加载状态字典
    model.load_state_dict(state_dict)
    # 设置为评估模式
    return model.eval()