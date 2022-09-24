import random
import sys
from collections import Counter
import pygame.gfxdraw
import pygame
import random

'''
author : 1218x
'''


class image:
    def __init__(self, x, y, image1,number,under,screen,i,j):
        self.x = x
        self.y = y
        self.number = number
        self.image = image1
        self.under = under
        self.screen = screen
        self.i=i
        self.j=j

    def color(self):
        pygame.gfxdraw.box(self.screen, pygame.Rect(self.x, self.y, 58, 62),(105,105,105,158))


def create_list():
    # 游戏代码...

    '''----------初始化所有列表----------'''

    #卡牌类别数
    typeNum = 13

    #卡牌总数
    cardNum = [[i] for i in range(typeNum*3*7)]

    #将卡牌打乱序号
    random.shuffle(cardNum)

    #每种卡牌21张
    numlist = [[i for i in range(13)]for i in range(21)]
    #化为一维列表
    numlist_new = []
    for i in numlist:
        for j in i:
            numlist_new.append(j)
    #以种类打乱卡牌
    numlist_shuffle = [0 for i in range(typeNum*3*7)]
    for index in range(len(cardNum)):
        numlist_shuffle[cardNum[index][0]] = numlist_new[index]
    numlist = numlist_shuffle

    #打乱后卡牌编号
    cardNumList = [i for i in range(typeNum * 3 * 7)]

    #卡槽和卡池列表
    storeList=[]
    cardList=[]
    cardMat = [[0 for i in range(7)] for i in range (typeNum*3*7)]

    for i in range(len(cardMat)):
        cardMat[i][0] = i+1
    return numlist, storeList, cardList


def refresh(numlist):
    num_new = []
    for i in range(273):
        if numlist[i] != -1:
            num_new.append(numlist[i])
    random.shuffle(num_new)
    n = len(num_new)
    x=0
    for i in range(273):
        if numlist[i] != -1:
            numlist[i] = num_new[x]
            x+=1
    return numlist


def main():
    pygame.init()
    numlist,storeList,cardList = create_list()

    '''--------开始作图---------'''

    # 创建游戏主窗口
    screen = pygame.display.set_mode((462,720))

    # 设置窗口的标题，即游戏名称
    pygame.display.set_caption('羊了个羊 v0.5')

    # 绘制背景图像
    # 1> 加载背景
    bg = pygame.image.load("./imd/bkg.jpg")
    bg_new = pygame.transform.scale(bg,(462,720))
    img = []
    img_new = []
    for i in range(1,14):
        img.append(pygame.image.load('./imd/card_'+str(i)+'.png'))
        img_new.append(pygame.transform.scale(img[-1], (58, 62)))
    re1 = pygame.image.load('./imd/re1.png')
    re2 = pygame.image.load('./imd/re2.png')
    store = pygame.image.load('./imd/store.jpg')
    re1 = pygame.transform.scale(re1,(70,60))
    re2 = pygame.transform.scale(re2,(70,60))
    store = pygame.transform.scale(store,(462,117))

    # 游戏循环
    k1, k2, k3, k4 = 224, 240, 256, 272
    deep = [[0,0,0,0,0]for i in range(4)]
    while True:
        # 设置窗口的标题，即游戏名称
        pygame.display.set_caption('羊了个羊 v0.5')

        # 绘制背景图像
        screen.blit(bg_new, (0, 0))
        screen.blit(re1, (20, 20))
        screen.blit(re2, (105, 20))
        screen.blit(store, (0, 580))
        for i in range(len(storeList)):
            screen.blit(img_new[storeList[i]], (24 + 59 * i, 602))
        #开始绘制卡牌,并记录卡牌坐标
        img_list=[]
        def draw():
            for k in range(5):
                for i in range(3):
                    for j in range(3):
                        if numlist[k*9+i*3+j]!=-1:
                            screen.blit(img_new[numlist[k*9+i*3+j]], (148+58*j, 129+62*i))
                            if j == 3 and (deep[i][j + 1] == 8.5):
                                deep[i][j+1]+=0.5
                            if j==0 and (deep[i][j]==8.5):
                                deep[i][0]+=0.5
                            img_list.append(image(i=[i],j=[j+1],screen=screen,x=148+58*j,y=129+62*i,number=k*9+i*3+j,
                                                  under=13-deep[i][j+1]-k,image1=img_new[numlist[k*9+i*3+j]]))
                            if img_list[-1].under>0:
                                img_list[-1].color()
            for k in range(2):
                for i in range(4):
                    for j in range(4):
                        if numlist[45+k * 16 + i * 4 + j] != -1:
                            screen.blit(img_new[numlist[45+k * 16 + i * 4 + j]], (119+58*j, 129+62*i))
                            if j==0 and (deep[i][j]==7.5 or deep[i][j]==8.5):
                                deep[i][0]+=0.5
                            if j==3 and (deep[i][j+1]==7.5 or deep[i][j+1]==8.5):
                                deep[i][4]+=0.5
                            img_list.append(
                                image(i=[i],j=[j,j+1],screen=screen,x=119+58*j, y= 129+62*i, number=45+k * 16 + i * 4 + j,
                                      under=1 -min(deep[i][j]-7,deep[i][j+1]-7)*2- k,image1=img_new[numlist[45+k * 16 + i * 4 + j]]))
                            if img_list[-1].under>0:
                                img_list[-1].color()
            for k in range(2):
                for i in range(4):
                    for j in range(5):
                        if numlist[77+k * 20 + i * 5 + j] != -1:
                            screen.blit(img_new[numlist[77+k * 20 + i * 5 + j]], (90+58*j, 129+62*i))
                            if j == 3 and (deep[i][j + 1] == 4.5):
                                deep[i][j+1]+=0.5
                            if j==0 and (deep[i][j]==4.5):
                                deep[i][0]+=0.5
                            img_list.append(
                                image(i=[i],j=[j],screen=screen,x=90+58*j, y=129 + 62 * i, number=77+k * 20 + i * 5 + j, under=6-deep[i][j] - k,image1=img_new[numlist[77+k * 20 + i * 5 + j]]))
                            if img_list[-1].under>0:
                                img_list[-1].color()
            for k in range(2):
                for i in range(4):
                    for j in range(4):
                        if numlist[117+k * 16 + i * 4 + j] != -1:
                            screen.blit(img_new[numlist[117+k * 16 + i * 4 + j]], (119+58*j, 129+62*i))
                            if j==0 and (deep[i][j]==3.5 or deep[i][j]==4.5):
                                deep[i][0]+=0.5
                            if j==3 and (deep[i][j+1]==3.5 or deep[i][j+1]==4.5):
                                deep[i][4]+=0.5
                            img_list.append(
                                image(i=[i],j=[j,j+1],screen=screen,x=119+58*j, y= 129+62*i, number=117+k * 16 + i * 4 + j,
                                      under=1 -min(deep[i][j]-3,deep[i][j+1]-3)*2- k,image1=img_new[numlist[117+k * 16 + i * 4 + j]]))
                            if img_list[-1].under>0:
                                img_list[-1].color()
            for k in range(3):
                for i in range(4):
                    for j in range(5):
                        if numlist[149+k * 20 + i * 5 + j] != -1:
                            screen.blit(img_new[numlist[149+k * 20 + i * 5 + j]], (90+58*j, 129+62*i))
                            img_list.append(
                                image(i=[i],j=[j],screen=screen,x=90+58*j, y=129 + 62 * i, number=149+k * 20 + i * 5 + j, under=2-deep[i][j] - k,image1=img_new[numlist[149+k*20+i*5+j]]))
                            if img_list[-1].under>0:
                                img_list[-1].color()
            for i in range(209,225):
                if numlist[i] != -1:
                    screen.blit(img_new[numlist[i]], (30+7*(i-209), 420))
                    img_list.append(
                        image(i=[i],j=[5],x=30+7*(i-209), y=420, number=i, screen=screen,under=k1-i,image1=img_new[numlist[i]]))
                    if img_list[-1].under > 0:
                        img_list[-1].color()
            for i in range(225,241):
                if numlist[i] != -1:
                    screen.blit(img_new[numlist[i]], (372-7*(i-225), 420))
                    img_list.append(
                        image(i=[i],j=[5],x=372-7*(i-225), y=420, number=i, under=k2- i,screen=screen,image1=img_new[numlist[i]]))
                    if img_list[-1].under > 0:
                        img_list[-1].color()
            for i in range(241,257):
                if numlist[i] != -1:
                    screen.blit(img_new[numlist[i]], (30+7*(i-241), 500))
                    img_list.append(
                        image(i=[i],j=[5],x=30+7*(i-241), y=500, number=i, under=k3 - i,screen=screen,image1=img_new[numlist[i]]))
                    if img_list[-1].under > 0:
                        img_list[-1].color()
            for i in range(257,273):
                if numlist[i] != -1:
                    screen.blit(img_new[numlist[i]], (372-7*(i-257), 500))
                    img_list.append(
                        image(i=[i],j=[5],x=372-7*(i-257), y=500, number=i, under=k4 - i,screen=screen,image1=img_new[numlist[i]]))
                    if img_list[-1].under > 0:
                        img_list[-1].color()
            # 3> 更新显示
            pygame.display.update()
        draw()
        # 事件监听
        for event in pygame.event.get():

            # 判断用户是否点击了关闭按钮
            if event.type == pygame.QUIT:
                print("退出游戏...")

                pygame.quit()

                # 直接退出系统
                exit()

            #判断鼠标点击事件
            if event.type == pygame.MOUSEBUTTONDOWN:
                x,y = event.pos
                if x>20 and x<90 and y>20 and y<80:
                    main()
                elif x>105 and x<175 and y>20 and y<80:
                    numlist = refresh(numlist)
                else:
                    print(1)
                    for item in img_list:
                        if x > item.x and (x < item.x + 58) and y > item.y and (y < item.y + 62) and item.under <= 0:
                            print(2)
                            storeList.append(numlist[item.number])
                            numlist[item.number]=-1
                            i, j = item.i, item.j
                            if max(i)<4 and max(j)<5:
                                if len(i)==1 and len(j)==1:
                                    deep[i[0]][j[0]]+=1
                                if len(i)==1 and len(j)>1:
                                    deep[i[0]][j[0]] += 0.5
                                    deep[i[0]][j[1]] += 0.5
                                if len(i)>1 and len(j)==1:
                                    deep[i[0]][j[0]] += 0.5
                                    deep[i[1]][j[0]] += 0.5
                            if item.number>=209:
                                if item.number<=224:
                                    k1=k1-1
                                    break
                                if item.number<=240:
                                    k2=k2-1
                                    break
                                if item.number<=256:
                                    k3=k3-1
                                    break
                                if item.number<=274:
                                    k4=k4-1
                                    break
                            break
            storeList.sort()
            x = dict(Counter(storeList))
            lis=[]
            if len(storeList)>=7:
                if max(x.values()) < 3 :
                    print("lose!")
                    main()
            for i,j in x.items():
                if j>=3:
                    lis.append(i)
            for i in lis:
                storeList.remove(i)
                storeList.remove(i)
                storeList.remove(i)
            # 3> 更新显示
            pygame.display.update()
            flag=0
            for i in numlist:
                if i!=-1:
                    flag=1
            if flag==0:
                print("游戏结束，成功加入羊群！")
    pygame.quit()


if __name__=='__main__':
    main()
