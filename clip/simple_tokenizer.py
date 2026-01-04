# 导入必要的模块
import gzip  # 用于读取压缩的词汇表文件
import html  # 用于HTML实体解码（如 &amp; -> &）
import os  # 操作系统接口，用于文件路径操作
from functools import lru_cache  # 缓存装饰器，提高函数性能

import ftfy  # 修复文本的工具（Fix Text For You），处理编码问题
import regex as re  # 增强版正则表达式库，支持Unicode属性


# 使用LRU缓存装饰器，缓存函数结果，避免重复计算
# 获取默认BPE词汇表文件路径的函数
@lru_cache()
def default_bpe():
    # 返回词汇表文件的绝对路径
    # __file__是当前文件路径
    # os.path.dirname获取目录路径
    # os.path.abspath获取绝对路径
    # os.path.join拼接路径，得到bpe_simple_vocab_16e6.txt.gz的完整路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


# 创建字节到Unicode字符映射表的函数
@lru_cache()  # 缓存结果，这个映射表是固定的
def bytes_to_unicode():
    """
    返回utf-8字节列表和对应的Unicode字符串列表。
    可逆的BPE编码基于Unicode字符串工作。
    这意味着如果你想要避免UNK（未知标记），你的词汇表需要包含大量Unicode字符。
    当处理约100亿标记的数据集时，你需要大约5000个字符来获得良好的覆盖。
    这对于通常的32K BPE词汇表来说是一个很大的比例。
    为了避免这种情况，我们创建utf-8字节和Unicode字符串之间的查找表。
    并且避免映射到BPE编码无法处理的空白/控制字符。
    """
    # bs: 字节值列表，包含所有可打印字符的字节值
    # ord("!")到ord("~"): ASCII可打印字符（33-126）
    # ord("¡")到ord("¬"): Latin-1补充字符（161-172）
    # ord("®")到ord("ÿ"): Latin-1补充字符（174-255）
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))

    cs = bs[:]  # 初始cs与bs相同
    n = 0

    # 遍历所有256个字节值
    for b in range(2 ** 8):  # 2**8 = 256
        if b not in bs:  # 如果字节不在可打印字符中
            bs.append(b)  # 添加到字节列表
            cs.append(2 ** 8 + n)  # 映射到256以上的Unicode码点
            n += 1

    cs = [chr(n) for n in cs]  # 将码点转换为Unicode字符
    return dict(zip(bs, cs))  # 返回字节到Unicode字符的映射字典


# 获取单词中符号对的函数
def get_pairs(word):
    """返回单词中符号对的集合。
    单词表示为符号元组（符号是可变长度的字符串）。
    """
    pairs = set()  # 使用集合避免重复
    prev_char = word[0]  # 前一个字符

    # 遍历单词中的每个字符
    for char in word[1:]:
        pairs.add((prev_char, char))  # 添加相邻字符对
        prev_char = char  # 更新前一个字符

    return pairs  # 返回所有相邻字符对的集合


# 基本文本清洗函数
def basic_clean(text):
    # 使用ftfy修复常见文本问题（编码错误、乱码等）
    text = ftfy.fix_text(text)
    # 双重unescape：处理HTML实体（如 &lt; -> <, &amp; -> &）
    # 双重调用是为了处理嵌套的HTML实体
    text = html.unescape(html.unescape(text))
    return text.strip()  # 去除首尾空白


# 空白字符清洗函数
def whitespace_clean(text):
    # 将所有空白字符（空格、制表符、换行等）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()  # 去除首尾空格


# 简单的分词器类
class SimpleTokenizer(object):
    # 初始化分词器
    # bpe_path: BPE词汇表文件路径，默认为default_bpe()返回的路径
    def __init__(self, bpe_path: str = default_bpe()):
        # 字节编码器：将字节映射到Unicode字符
        self.byte_encoder = bytes_to_unicode()
        # 字节解码器：将Unicode字符映射回字节
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 读取BPE合并规则
        # 1. 用gzip打开压缩文件
        # 2. 读取内容
        # 3. 解码为UTF-8字符串
        # 4. 按行分割
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')

        # 提取有效的合并规则（跳过第一行说明和最后两行特殊标记）
        # 49152-256-2+1 = 49152是总词汇表大小，256是字节数，2是特殊标记
        merges = merges[1:49152 - 256 - 2 + 1]

        # 将每行的两个符号分割为元组
        # 例如："a b" -> ("a", "b")
        merges = [tuple(merge.split()) for merge in merges]

        # 构建词汇表
        vocab = list(bytes_to_unicode().values())  # 基本Unicode字符
        vocab = vocab + [v + '</w>' for v in vocab]  # 添加单词结束标记

        # 添加BPE合并后的符号
        for merge in merges:
            vocab.append(''.join(merge))  # 将合并符号连接起来

        # 添加特殊标记
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # 创建编码器和解码器
        self.encoder = dict(zip(vocab, range(len(vocab))))  # 符号->ID
        self.decoder = {v: k for k, v in self.encoder.items()}  # ID->符号

        # 创建BPE合并排序（rank）字典，用于决定合并优先级
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        # 缓存已经处理过的token，提高效率
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # 编译正则表达式模式，用于初步分词
        # 匹配：开始标记|结束标记|'s|'t|'re|'ve|'m|'ll|'d|字母+|数字|非空白非字母数字字符
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE
        )

    # BPE算法实现：将token分解为BPE子词
    # token: 输入的token字符串
    def bpe(self, token):
        # 检查缓存
        if token in self.cache:
            return self.cache[token]

        # 在token末尾添加单词结束标记</w>
        # 将字符串转换为元组以便处理
        word = tuple(token[:-1]) + (token[-1] + '</w>',)

        # 获取所有可能的字符对
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回带结束标记的token
        if not pairs:
            return token + '</w>'

        # 不断合并最高优先级的字符对
        while True:
            # 找到优先级最高的字符对（在bpe_ranks中rank值最小的）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))

            # 如果这个字符对不在合并规则中，停止合并
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram  # 拆分为两个字符
            new_word = []  # 存储合并后的新单词
            i = 0  # 当前位置索引

            # 遍历原单词，寻找并合并指定的字符对
            while i < len(word):
                try:
                    # 查找第一个字符出现的位置
                    j = word.index(first, i)
                    # 将当前位置到j位置的字符添加到新单词
                    new_word.extend(word[i:j])
                    i = j  # 更新位置
                except:
                    # 如果没有找到，添加剩余字符并退出循环
                    new_word.extend(word[i:])
                    break

                # 如果找到了字符对(first, second)
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 合并两个字符
                    i += 2  # 跳过两个字符
                else:
                    new_word.append(word[i])  # 只添加第一个字符
                    i += 1

            # 转换为元组形式
            new_word = tuple(new_word)
            word = new_word  # 更新单词

            # 如果单词只剩下一个符号，停止合并
            if len(word) == 1:
                break
            else:
                # 重新计算字符对
                pairs = get_pairs(word)

        # 将元组连接为字符串，符号间用空格分隔
        word = ' '.join(word)

        # 缓存结果
        self.cache[token] = word

        return word

    # 编码函数：将文本转换为token ID列表
    # text: 输入的文本字符串
    def encode(self, text):
        bpe_tokens = []  # 存储最终的token IDs

        # 文本预处理：清洗、去HTML实体、标准化空白
        text = whitespace_clean(basic_clean(text)).lower()

        # 使用正则表达式进行初步分词
        for token in re.findall(self.pat, text):
            # 将token的每个字节编码为Unicode字符
            # 这是为了解决BPE处理字节而不是字符的问题
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))

            # 对token进行BPE编码，然后分割为子词
            # 将每个子词转换为对应的ID并添加到结果列表
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')
            )

        return bpe_tokens  # 返回token ID列表

    # 解码函数：将token ID列表转换回文本
    # tokens: token ID列表
    def decode(self, tokens):
        # 将token IDs转换回符号字符串
        text = ''.join([self.decoder[token] for token in tokens])

        # 处理解码结果：
        # 1. 将Unicode字符转换回字节
        # 2. 解码为UTF-8字符串
        # 3. 将单词结束标记</w>替换为空格
        # errors="replace": 遇到解码错误时用替换字符(�)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')

        return text  # 返回解码后的文本