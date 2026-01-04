import pickle  # 导入pickle模块，用于加载序列化数据
import pprint  # 导入美观打印模块，可以自动截断长数据并添加省略号
# 打开数据集文件
import numpy as np


path = "flickr25k.pkl"
# path = "coco2017.pkl"
with open(path, "rb") as f:  # 二进制读模式
    data = pickle.load(f)  # 加载数据

print("=" * 60)
print(f"数据类型: {type(data)}")  # 显示是字典、列表等


# 创建pprint对象，设置参数
pp = pprint.PrettyPrinter(
    indent=2,           # 缩进2个空格
    width=80,           # 每行最大宽度80字符
    depth=1,            # 最大嵌套深度1层（超过会显示...）
    compact=True,       # 紧凑模式
    sort_dicts=False    # 不排序字典键（保持原始顺序）
)

# 打印数据 - pprint会自动处理长数据，添加省略号
print("=" * 10 + "使用pprint美观打印（会自动添加省略号）:" + "=" * 10)
pp.pprint(data)  # 这是关键！会自动截断过长的列表/字典
# 字典
print("data.keys()",data.keys())
print("indexs.length",len(data["indexs"]))
print("captions.length",len(data["captions"]))
print("labels.length",len(data["labels"]))
print(data["labels"][0].shape)
# 打印具体内容
print("=" * 10 + "打印具体内容" + "=" * 10)
print(data["indexs"][0:2])
print(data["captions"][0:2])
data["labels"] = [arr.astype(np.int8) for arr in data["labels"]]
print(data["labels"][0:2])

