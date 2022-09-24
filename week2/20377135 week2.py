import csv
import math

import jieba
from collections import Counter
import random
import pandas as pd
import os
from os import path
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec
import numpy as np
from scipy import spatial

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('C:\\Users\\dexter\\Desktop\\stopwords_list.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# 获取当前文件路径
def wordvision(data):
    #d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    # 获取文本txt的路径（txt和代码在一个路径下面）
    #text = open(path.join(d,path),encoding='UTF-8').read()

    # 生成词云
    wc = WordCloud(
            font_path='c:\windows\Fonts\simhei.ttf',
            scale=2,
            max_font_size=100,  #最大字号
            background_color='white',  #设置背景颜色
            max_words=50
            )

    wc.generate_from_frequencies(data)  # 从文本生成wordcloud
    # wc.generate_from_text(text)  #用这种表达方式也可以

    # 显示图像
    plt.imshow(wc,interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    wc.to_file('C:\\Users\\dexter\\Desktop\\num_visual.jpg')  # 储存图像
    #plt.savefig('C:\\Users\\dexter\\Desktop\\num_visual.jpg',dpi=200)  #用这个可以指定像素
    plt.show()


# WordCount
def worldcount(content):
    with open('C:\\Users\\dexter\\Desktop\\output.txt', 'r', encoding='UTF-8') as fr:  # 读入已经去除停用词的文件
        data = jieba.cut(fr.read())
    data = dict(Counter(data))

    high = []
    with open('C:\\Users\\dexter\\Desktop\\cipin.txt', 'w', encoding='UTF-8') as fw:  # 读入存储wordcount的文件路径
        for k, v in data.items():
            fw.write('%s,%d\n' % (k, v))
            if v >= 5:
                high.append(k)
    n = len(high)
    sum = 0
    one_hot = [[0 for i in range(len(high))] for i in range(len(content))]
    for i in range(len(content)):
        sum += len(content[i].split())
        for j in range(len(high)):
            if high[j] in content[i]:
                one_hot[i][j] += 1
    return high, one_hot,data


def onehot(content,high,one_hot):
    list = range(len(content))
    num = random.sample(list, 10)
    dis=[[0 for i in range(10)]for i in range(10)]
    ans=0
    for i in range(10):
        for j in range(10):
            for k in range(len(high)):
                ans+=(one_hot[num[i]][k]-one_hot[num[j]][k])*(one_hot[num[i]][k]-one_hot[num[j]][k])
            dis[i][j]=math.sqrt(ans)
            ans=0
    for i in range(10):
        print(dis[i])

def tfidf(content,high,one_hot):
    count1=[0 for i in range(len(high))]
    count2=[0 for i in range(len(high))]
    n=0
    for j in range(len(high)):
        for i in range(len(content)):
            if j==0:
                n+=len(content[i])
            if one_hot[i][j]!=0:
                count1[j]+=one_hot[i][j]
                count2[j]+=1

    tf_idf = [0 for i in range(len(high))]
    for i in range(len(high)):
        tf_idf[i]= count1[i]/n * math.log(len(content)/(count2[j]+1))
        if tf_idf[i]>0.01:
            print('%s,%.2f\n' % (high[i], tf_idf[i]))

def word2vec_(path_):
    # 文件位置需要改为自己的存放路径
    # 加载语料
    sentences = word2vec.LineSentence(path_)
    # 训练语料
    path = get_tmpfile("word2vec.model")  # 创建临时文件
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=10)
    model.save("C:\\Users\\dexter\\Desktop\\word2vec.model")
    # model = word2vec.Word2Vec.load("word2vec.model") #第二次使用直接加载
    # 输出与“武汉”相近的50个词
    for key in model.wv.similar_by_word('武汉', topn=50):
        print(key)
    # 查看某个词的向量
    print(model.wv['武汉'])

def sim(path,a1,a2):
    model = word2vec.Word2Vec.load(path) #加载模型
    index_to_key = set(model.wv.index_to_key)

    def avg_feature_vector(sentence, model, num_features, index_to_key):
        words = sentence.split()    #将分词后的弹幕每个词分割开
        feature_vec = np.zeros((num_features, ), dtype='float32')    #初始化特征向量
        n_words = 0
        for word in words:
            if word in index_to_key:
                n_words += 1

                feature_vec = np.add(feature_vec, model.wv[word])    #将所有词特征向量相加
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)    #总和除词数得到平均值
        return feature_vec

    #计算两条弹幕的平均向量
    s1_afv = avg_feature_vector(a1, model=model, num_features=100, index_to_key=index_to_key)
    s2_afv = avg_feature_vector(a2, model=model, num_features=100, index_to_key=index_to_key)
    #计算余弦相似度
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    print(sim)

def main():
    inputs = open("C:\\Users\\dexter\\Desktop\\danmuku.csv",encoding='UTF-8') #加载要处理的文件的路径
    outputs = open('C:\\Users\\dexter\\Desktop\\output.txt', 'w',encoding='UTF-8') #加载处理后的文件路径
    reader=csv.reader(inputs)
    header_row=next(reader)
    content = []
    i=0
    for row in reader:
        i+=1
        content.append(row[0])
        if i>10000:
            break

    for item in content:
        line_seg = seg_sentence(item)
        outputs.write(line_seg+'\n')
    outputs.close()
    inputs.close()

    high,one_hot,data = worldcount(content)

    onehot(content,high,one_hot)
    tfidf(content,high,one_hot)
    wordvision(data)

if __name__ == '__main__':
    main()