import math
import pandas as pd
import numpy as np
import re
import jieba
import random
from matplotlib import pyplot as plt

import webbrowser
from pyecharts.charts import Geo
from pyecharts.globals import GeoType
from pyecharts import options as opts
from tqdm import tqdm

#情绪词加入初始字典
def load():
    jieba.load_userdict(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\anger.txt")
    jieba.load_userdict(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\disgust.txt")
    jieba.load_userdict(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\fear.txt")
    jieba.load_userdict(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\joy.txt")
    jieba.load_userdict(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\sadness.txt")

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='UTF-8').readlines()]
    return stopwords


def add_item():
    inputs = open("C:\\Users\\dexter\\Desktop\\weibo.txt",encoding='UTF-8') #加载要处理的文件的路径
    load()
    location,week,month,time,content,emo=[],[],[],[],[],[]
    i=0
    num = random.sample(range(2000000), 10000)
    pbar = tqdm(total=10000)
    for item in inputs:
        if i in num:
            location.append(str(findlocation(item)))
            week.append(str(findtime('week',item)))
            month.append(str(findtime('month',item)))
            time.append(str(findtime('time', item)))
            con=findcontent(item,str(findlocation(item)),str(findtime('week',item)))
            con = wash(str(con))
            con=seg_sentence(con)
            content.append(con)
            emo.append(emotionlist(con)[1][0])
            pbar.update(1)
        i+=1
    pbar.close()
    return location,week,month,time,content,emo

#提取位置信息
def findlocation(sentence):
    loc=[]
    for i in sentence:
        if i!=']':
            loc.append(i)
        if i==']':
            break
    loc.append(']')
    return ''.join(loc)

#提取时间信息
def findtime(model,sentence):
    weeklist=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    monthlist=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    if model=='week':
        for i in range(len(sentence)):
            if sentence[i:i+3] in weeklist:
                return sentence[i:i+3]
    if model=='month':
        for i in range(len(sentence)):
            if sentence[i:i+3] in monthlist:
                return sentence[i:i+3]
    if model=='time':
        for i in range(len(sentence)):
            if (sentence[i:i+3] in weeklist) and (sentence[i+4:i+7] in monthlist):
                return sentence[i+11:i+13]

#提取文本内容
def findcontent(sentence,location,week):
    sentence=sentence.replace(location,'')
    for i in range(len(sentence)):
        if sentence[i:i+3]==week:
            return sentence[:i]

#清洗数据，删除url
def wash(sentence):
    text = re.sub("https*\S+", " ", sentence)
    return text

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

#两种方式计算情绪向量
def emotionlist(sentence):
    #加载情绪词典，方便后续判断情绪
    angry = [line.strip() for line in open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\anger.txt", 'r',encoding='UTF-8').readlines()]
    disgust = [line.strip() for line in open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\disgust.txt", 'r',encoding='UTF-8').readlines()]
    fear = [line.strip() for line in open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\fear.txt", 'r',encoding='UTF-8').readlines()]
    joy =  [line.strip() for line in open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\joy.txt", 'r',encoding='UTF-8').readlines()]
    sadness = [line.strip() for line in open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\sadness.txt", 'r',encoding='UTF-8').readlines()]

    def vector_1(sentence):
        nonlocal angry, disgust, fear, joy, sadness
        word = sentence.split()
        num = ([0, 0, 0, 0, 0])
        for item in word:
            if item in angry:
                num[0] += 1
                continue
            if item in disgust:
                num[1] += 1
                continue
            if item in fear:
                num[2] += 1
                continue
            if item in joy:
                num[3] += 1
                continue
            if item in sadness:
                num[4] += 1
                continue
        n = sum(num)
        if n==0:
            return num
        return [num[0] / n,num[1] / n,num[2] / n,num[3] / n,num[4] / n]

    def vector_2(sentence):
        nonlocal angry, disgust, fear, joy, sadness
        word = sentence.split()
        num = [0, 0, 0, 0, 0]
        for item in word:
            if item in angry:
                num[0] += 1
                continue
            if item in disgust:
                num[1] += 1
                continue
            if item in fear:
                num[2] += 1
                continue
            if item in joy:
                num[3] += 1
                continue
            if item in sadness:
                num[4] += 1
                continue
        if max(num)==0:
            return [-1,0]
        return [np.argmax(num), max(num)]
    vector1=vector_1(sentence)
    vector2=vector_2(sentence)

    def add_word(sentence, lis1, lis2, lis3, lis4, lis5):
        nonlocal  vector1
        word = sentence.split()

    return [vector1,vector2]

#返回每句话分析所得情绪
def count_all(content):
    dic = {-1:'None',0: 'angry', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness'}
    emo=[]
    for i in content:
        emotion = emotionlist(i)
        index = emotion[1][0]
        emo.append(dic[index])
    return emo

#添加字典关键词
def add_word(sentence):
    #调用emotionlist,基于emotionlist结果再次遍历文本
    num = emotionlist(sentence)[1][0]
    lis1, lis2, lis3, lis4, lis5 = [],[],[],[],[]
    # 加载情绪词典，方便后续判断情绪
    angry = [line.strip() for line in open(
        r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\anger.txt", 'r',
        encoding='UTF-8').readlines()]
    disgust = [line.strip() for line in open(
        r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\disgust.txt", 'r',
        encoding='UTF-8').readlines()]
    fear = [line.strip() for line in
            open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\fear.txt",
                 'r', encoding='UTF-8').readlines()]
    joy = [line.strip() for line in
           open(r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\joy.txt",
                'r', encoding='UTF-8').readlines()]
    sadness = [line.strip() for line in open(
        r"C:\Users\dexter\Desktop\Anger makes fake news viral online-data&code\data\emotion_lexicon\sadness.txt", 'r',
        encoding='UTF-8').readlines()]

    word = sentence.split()
    for item in word:
        if (item not in angry) and (item not in disgust) and (item not in fear) and (item not in joy)\
                and (item not in sadness):
            if num==0:
                lis1.append(item)
                continue
            if num==1:
                lis2.append(item)
                continue
            if num==2:
                lis3.append(item)
                continue
            if num==3:
                lis4.append(item)
                continue
            if num==4:
                lis5.append(item)
    return lis1, lis2, lis3, lis4, lis5

def add_dic(content):
    dic1,dic2,dic3,dic4,dic5={},{},{},{},{}
    for sentence in content:
        lis1, lis2, lis3, lis4, lis5 = add_word(sentence)
        def add(lis,dic):
            for item in lis:
                if item not in dic:
                    dic[item]=1
                else:
                    dic[item]+=1
            return dic
        dic1 = add(lis1,dic1)
        dic2 = add(lis2,dic2)
        dic3 = add(lis3,dic3)
        dic4 = add(lis4,dic4)
        dic5 = add(lis5,dic5)
    def travel(dic):
        lis=[]
        for i,j in dic.items():
            if j>10:
                lis.append(i)
        return lis
    lis1 = travel(dic1)
    lis2 = travel(dic2)
    lis3 = travel(dic3)
    lis4 = travel(dic4)
    lis5 = travel(dic5)
    return lis1, lis2, lis3, lis4, lis5

def add_file(lis,filename):
    file = open(filename, 'a',encoding='UTF-8')
    for item in lis:
        file.write(item + '\n')
    file.close()

#计算每个时间段不同情绪分布
def count_emotion(content,time,model):
    if model=='time':
        count = [[0 for i in range(5)] for i in range(24)]
        for i in range(len(content)):
            emo=emotionlist(content[i])
            #dic={0:'angry',1:'disgust',2:'fear',3:'joy',4:'sadness'}
            if emo[1][0]==-1:
                continue
            count[int(time[i])][emo[1][0]]+=1
    if model=='week':
        count = [[0 for i in range(5)] for i in range(7)]
        for i in range(len(content)):
            emo=emotionlist(content[i])
            dic={'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4,'Sat':5,'Sun':6}
            if emo[1][0]==-1:
                continue
            count[dic[time[i]]][emo[1][0]]+=1
    if model=='month':
        count = [[0 for i in range(5)] for i in range(12)]
        for i in range(len(content)):
            emo=emotionlist(content[i])
            dic={'Jan':0,'Feb':1,'Mar':2,'Apr':3,'May':4,'Jun':5,'Jul':6,'Aug':7,'Sep':8,'Oct':9,'Nov':10,'Dec':11}
            if emo[1][0]==-1:
                continue
            count[dic[time[i]]][emo[1][0]]+=1
    #print(count)
    return count

#计算每个时间主要情绪及其占比
def time_emo(count):
    dic = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness'}
    cou=[[]for i in range(len(count))]
    for i in range(len(count)):
        index=np.argmax(count[i])
        if sum(count[i])==0:
            num=0
            cou[i]=['None',0]
        else:
            num=max(count[i])/sum(count[i])
            cou[i]=[dic[index],num]
    return cou

#将位置信息转化为列表形式
def loc_lis(location):
    lis_loc=[]
    for i in location:
        lis_loc.append(i[1:-1].split(', '))
    return lis_loc

#计算随着与中心的距离变化的情绪分布
def loc_dis(loc,dis,emo):
    #计算中心
    x,y=0,0
    for i in loc:
        x+=float(i[0])
        y+=float(i[1])
    #print(x/len(loc),y/len(loc))
    def count_dis(dis):
        nonlocal x,y,loc,emo

        lis=np.array([0, 0, 0, 0, 0])
        for i in range(len(loc)):
            if math.sqrt((float(loc[i][0])-x/len(loc))**2+(float(loc[i][1])-y/len(loc))**2) < dis and emo[i]!=-1:
                lis[emo[i]]+=1
        if sum(lis[i] for i in range(5))==0:
            return lis
        return lis/sum(lis[i] for i in range(5))
    return count_dis(dis)

#绘制随时间变化情绪分布折线图
def plot_time(count_emo,model,i):
    if model=='week':
        time = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    if model == 'time':
        time = [i for i in range(24)]

    if model == 'month':
        time = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    angry = [i[0] for i in count_emo]
    disgust = [i[1] for i in count_emo]
    fear = [i[2] for i in count_emo]
    joy = [i[3] for i in count_emo]
    sadness = [i[4] for i in count_emo]
    plt.subplot(2,3,i)

    plt.xlabel(model)  # x轴名称 只能是英文
    plt.ylabel("frequency")  # y轴名称 只能是英文
    plt.title("emotion in different "+ model , fontdict={"fontsize": 16})  # 添加标题，调整字符大小

    plt.plot(time, angry, 'r', label="angry")
    plt.plot(time, disgust, 'm', label="disgust")
    plt.plot(time, fear, 'k', label="fear")
    plt.plot(time, joy, 'g', label="joy")
    plt.plot(time, sadness, 'b', label="sadness")

    plt.legend()

#绘制随与中心的距离变化情绪占比的变化曲线图
def plot_dis(loc,emo,n):
    x = np.linspace(0, 5, 500)
    angry,disgust,fear,joy,sadness=[],[],[],[],[]
    for i in x:
        y = loc_dis(loc,i,emo)
        angry.append(y[0])
        disgust.append((y[1]))
        fear.append(y[2])
        joy.append(y[3])
        sadness.append(y[4])

    plt.subplot(2,3,n)
    plt.xlabel("distance")  # x轴名称 只能是英文
    plt.ylabel("percent")  # y轴名称 只能是英文
    plt.title("emotion with different distance", fontdict={"fontsize": 16})  # 添加标题，调整字符大小
    plt.plot(x, angry,'r',label="angry")
    plt.plot(x, disgust,'m',label="disgust")
    plt.plot(x, fear,'k',label="fear")
    plt.plot(x,joy,'g',label="joy")
    plt.plot(x, sadness,'b',label="sadness")

    plt.legend()

def test_geo(emo,loc):
    g = (
        Geo(
        init_opts=opts.InitOpts(width="600px", height="600px", page_title="map", bg_color='#73B0E2')
        # 颜色是str的16进制或英文都可以
    ).add_schema(
        maptype="北京",
        itemstyle_opts=opts.ItemStyleOpts(
            color="#97C0E3"  # 背景颜色
            , border_color="black")
        # 地图类型
         # 边界线颜色
    )
    )
    #lis1,lis2,lis3,lis4,lis5=[],[],[],[],[]
    dic = {"angry":'#DADADA',"disgust":'#BBCEDF','fear':'#9ABFDE','joy':'#4999DC','sadness':'#1381DC'}
    for i in range(len(emo)):
        if emo[i]=='None':
            continue
        g.add_coordinate(emo[i], loc[i][1], loc[i][0])
        g.add(
            series_name=i  # 系列名
            , data_pair=[(emo[i],i)]  # 系列里需要的点用列表框住多个元组达到批量输入的效果[(坐标点1，坐标点1的值),(坐标点2，坐标点2的值),(坐标点3，坐标点3的值)]
            , symbol_size=5  # 系列内显示点的大小
            , color=dic[emo[i]]
            , is_selected=True
        )

    g.set_series_opts(
        label_opts=opts.LabelOpts(
            is_show=False
        ))
    g.set_global_opts(
        title_opts=opts.TitleOpts(
            title='微博情绪地区分布图',  # 主标题内容
            subtitle='————以北京为例',  # 副标题内容
            item_gap=15,  # 主副标题的间距
            title_textstyle_opts=opts.TextStyleOpts(
                color="white",  # 主标题颜色
                font_weight="bolder",  # 主标题加粗
                font_size=24  # 主标题字体大小
            ),
            subtitle_textstyle_opts=opts.TextStyleOpts(
                color='white',  # 副标题颜色
                font_weight="bolder",  # 副标题加粗
                font_size=15))  # 副标题副标题字体大小
        , legend_opts=opts.LegendOpts(pos_right="10px", inactive_color="white",
                                      textstyle_opts=opts.TextStyleOpts(color="orange"))
    )
    result = g.render()  # 会在你py文件同目录下生成一个html文件，也可在括号内输入保存路径，用浏览器打开html文件可以查看
    webbrowser.open(result)

def main():
    location,week,month,time,content,emo = add_item()
    count_emo_t = count_emotion(content,time,'time')
    count_emo_w = count_emotion(content, week, 'week')
    count_emo_m = count_emotion(content, month, 'month')

    result_t = time_emo(count_emo_t)
    result_w = time_emo(count_emo_w)
    result_m = time_emo(count_emo_m)

    lis1, lis2, lis3, lis4, lis5 = add_dic(content)
    print(lis1)
    print(lis2)
    print(lis3)
    print(lis4)
    print(lis5)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    plot_time(count_emo_t, 'time', 1)
    plot_time(count_emo_w, 'week', 2)
    plot_time(count_emo_m, 'month', 3)
    loc = loc_lis(location)
    plot_dis(loc, emo, 5)
    plt.show()

    emo_ = count_all(content)
    test_geo(emo_, loc)


if __name__=='__main__':
    main()
