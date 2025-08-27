import os
import jieba
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import numpy as np

# 读取评论数据
df = pd.read_csv('data/all_comments.csv')
# 获取评论内容
content_list = df['评论内容'].astype(str).to_list()
# 把列表转成字符串
content = ''.join(content_list)
# jieba分词
string = ' '.join(jieba.lcut(content))   # 合并成字符串
# print(string)

# 读取停用词表
stopwords = set()
try:
    with open('static/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去掉首尾的空白字符
            if line:  # 确保不是空行
                stopwords.add(line)
except Exception as e:
    print(f"Error reading stopwords.txt: {e}")


# # 读取形状图片并转换为 NumPy 数组
mask_image = Image.open('static/shape.png')  # 替换为你的形状图片路径
mask_array = np.array(mask_image)

# 词云图配置
wc = WordCloud(
    width=1000,
    height=800,
    background_color='white',
    font_path='msyh.ttc',  # 确保字体路径正确
    stopwords=stopwords,
    mask=mask_array,  # 使用形状图片作为 mask
    scale=25
)

# 传入文字内容
wc.generate(string)

# 输出词云图到 static 文件夹
output_path = os.path.join('static', "wordcloud.png")
wc.to_file(output_path)
