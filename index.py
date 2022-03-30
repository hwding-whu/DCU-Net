import cv2
from matplotlib import pyplot as plt
from numpy import *
import numpy as np

Dir_G = './test3/ground/'
Dir_P = './test3/prediction/'


# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:#Ground Truth和Prediction同为白色的部分，即被分割物品预测正确的部分。
                DSI_s += 1
            if binary_GT[i][j] == 255:#Ground Truth为白色的部分
                DSI_t += 1
            if binary_R[i][j] == 255:#Prediction为白色的部分
                DSI_t += 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall

def index():
    List_Dice = []
    List_Voe = []
    List_Rvd = []
    List_Pre = []
    List_Rec = []


    ListDice = []
    ListPre = []
    ListRec = []
    ListAcc = []
    for i in range(20):
        img_GT = cv2.imread(Dir_G+'%d.png'%(i), 0)
        img_R = cv2.imread(Dir_P+'%d.png'%(i), 0)
        ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        TP, FP, TN, FN = 0, 0, 0, 0
        row, col = binary_GT.shape  # 矩阵的行与列
        for j in range(row):
            for k in range(col):
                if img_GT[j][k] == 255 and img_R[j][k] == 255:#Ground Truth和Prediction物体重合的部分
                    TP = TP + 1
                if img_GT[j][k] == 0 and img_R[j][k] == 255:#Ground Truth为背景Prediction为物体
                    FP = FP + 1
                if img_GT[j][k] == 0 and img_R[j][k] == 0:
                    TN = TN + 1
                if img_GT[j][k] == 255 and img_R[j][k] == 0:
                    FN = FN + 1
        Dice = 2*TP/(2*TP + FP + FN)
        Pre = TP/(TP + FP)
        Rec = TP/(TP + FN)
        Acc = (TP + TN)/(TP + TN + FP + FN)
        ListDice.append(Dice)
        ListPre.append(Pre)
        ListRec.append(Rec)
        ListAcc.append(Acc)

        List_Dice.append(calDSI(binary_GT, binary_R))
        List_Voe.append(calVOE(binary_GT, binary_R))
        List_Rvd.append(calRVD(binary_GT, binary_R))
        List_Pre.append(calPrecision(binary_GT, binary_R))
        List_Rec.append(calRecall(binary_GT, binary_R))
    return List_Dice,List_Voe,List_Rvd,List_Pre,List_Rec,ListDice,ListPre,ListRec,ListAcc

if __name__ == '__main__':
    # step 1：读入图像，并灰度化
    List_Dice = []
    List_Voe = []
    List_Rvd = []
    List_Pre = []
    List_Rec = []

    ListDice = []
    ListPre = []
    ListRec = []
    ListAcc = []
    List_Dice, List_Voe, List_Rvd, List_Pre, List_Rec, ListDice, ListPre, ListRec, ListAcc=index()
    print ('Dice:',mean(List_Dice))
    print('Dice:', mean(ListDice))
    print('Pre:', mean(List_Pre))
    print('Pre:', mean(ListPre))
    print('Rec:', mean(List_Rec))
    print('Rec:', mean(ListRec))
    print('Acc:', mean(ListAcc))





'''
    img_GT = cv2.imread('01-G.png', 0)
    img_R = cv2.imread('01-P.png', 0)
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # 灰度化
    # img_GT = img_GT[:,:,[2, 1, 0]]
    # img_R  = img_R[:,: [2, 1, 0]]

    # step2：二值化
    # 利用大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # step 3： 显示二值化后的分割图像与真值图像
    plt.figure()
    plt.subplot(121), plt.imshow(binary_GT), plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(122), plt.imshow(binary_R), plt.title('Prediction')
    plt.axis('off')
    plt.show()

    # step 4：计算DSI
    print('（1）DICE计算结果，      DSI       = {0:.4}'.format(calDSI(binary_GT, binary_R)))  # 保留四位有效数字

    # step 5：计算VOE
    print('（2）VOE计算结果，       VOE       = {0:.4}'.format(calVOE(binary_GT, binary_R)))

    # step 6：计算RVD
    print('（3）RVD计算结果，       RVD       = {0:.4}'.format(calRVD(binary_GT, binary_R)))

    # step 7：计算Precision
    print('（4）Precision计算结果， Precision = {0:.4}'.format(calPrecision(binary_GT, binary_R)))

    # step 8：计算Recall
    print('（5）Recall计算结果，    Recall    = {0:.4}'.format(calRecall(binary_GT, binary_R)))
'''