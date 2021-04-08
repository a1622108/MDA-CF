# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/home/Daiqiuyin')
import time
import numpy as np
import pandas as pd
import csv
import math
import random
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
from math import e,log
from numpy import interp
from itertools import cycle
from gcforest.gcforest import GCForest
import pickle
from sklearn.ensemble import RandomForestClassifier
import os           # https://blog.keras.io/building-autoencoders-in-keras.html
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 只显示WARNING（警告）、ERROR（错误）
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras import regularizers

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, errors='ignore'))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

def FeatureGenerate(X_train,y_train,X_test):       
    AllRNA=[]
    ReadMyCsv(AllRNA, "/home/Daiqiuyin/D3数据集/data/AllRNA.csv")   
    AllDisease=[]
    ReadMyCsv(AllDisease, "/home/Daiqiuyin/D3数据集/data/AllDisease.csv")
    Famsim=[]
    ReadMyCsv2(Famsim, "/home/Daiqiuyin/D3数据集/data/Famsim.csv")
    SeqSim=[]
    ReadMyCsv2(SeqSim, "/home/Daiqiuyin/D3数据集/data/SeqSim2.csv")
    FuncSim=[]
    ReadMyCsv2(FuncSim, "/home/Daiqiuyin/D3数据集/data/FuncSim-gene.csv")
    DiseaseSimilarityModel1=[]
    ReadMyCsv2(DiseaseSimilarityModel1, "/home/Daiqiuyin/D3数据集/data/DiseaseSimilarityModel1.csv")
    DiseaseSimilarityModel2=[]
    ReadMyCsv2(DiseaseSimilarityModel2, "/home/Daiqiuyin/D3数据集/data/DiseaseSimilarityModel2.csv")    
    
    # 由rna-disease生成对应关系矩阵，有关系1，没关系0，行为疾病AllDisease，列为rna AllRNA
    # 生成全0矩阵
    DiseaseAndRNABinary = []
    counter = 0
    while counter < len(AllDisease):
        row = []
        counter1 = 0
        while counter1 < len(AllRNA):
            row.append(0)
            counter1 = counter1 + 1
        DiseaseAndRNABinary.append(row)
        counter = counter + 1

    counter = 0
    while counter < len(X_train):
        DN = X_train[counter][1]     # disease name
        RN = X_train[counter][0]     # rna name
        if y_train[counter]==1:
            counter1=0
            while counter1< len(AllDisease):
                if AllDisease[counter1][0]==DN:
                    counter2=0
                    while counter2< len(AllRNA):
                        if AllRNA[counter2][0]==RN:
                            DiseaseAndRNABinary[counter1][counter2]=1
                            break
                        counter2=counter2+1
                    break
                counter1=counter1+1
        counter=counter+1                       
    
    # 计算rd
    counter1 = 0
    sum1 = 0
    while counter1 < (len(AllDisease)):
        counter2 = 0
        while counter2 < (len(AllRNA)):
            sum1 = sum1 + pow((DiseaseAndRNABinary[counter1][counter2]), 2)
            counter2 = counter2 + 1
        counter1 = counter1 + 1
    Ak = sum1
    Nd = len(AllDisease)
    rdpie = 1
    rd = rdpie /(Ak / Nd)

    # 生成DiseaseGaussian
    DiseaseGaussian = []
    counter1 = 0
    while counter1 < len(AllDisease):#计算疾病counter1和counter2之间的similarity
        counter2 = 0
        DiseaseGaussianRow = []
        while counter2 < len(AllDisease):# 计算Ai*和Bj*
            AiMinusBj = 0
            sum2 = 0
            counter3 = 0
            AsimilarityB = 0
            while counter3 < len(AllRNA):#疾病的每个属性分量
                sum2 = pow((DiseaseAndRNABinary[counter1][counter3] - DiseaseAndRNABinary[counter2][counter3]), 2)#计算平方
                AiMinusBj = AiMinusBj + sum2
                counter3 = counter3 + 1
            AsimilarityB = math.exp(- (AiMinusBj*rd))
            DiseaseGaussianRow.append(AsimilarityB)
            counter2 = counter2 + 1
        DiseaseGaussian.append(DiseaseGaussianRow)
        counter1 = counter1 + 1
    
    # 构建RNAGaussian    
    MDiseaseAndRNABinary = np.array(DiseaseAndRNABinary)    # 列表转为矩阵
    RNAAndDiseaseBinary = MDiseaseAndRNABinary.T    # 转置
    RNAGaussian = []
    counter1 = 0
    sum1 = 0
    while counter1 < (len(AllRNA)):     # rna数量
        counter2 = 0
        while counter2 < (len(AllDisease)):     # disease数量
            sum1 = sum1 + pow((RNAAndDiseaseBinary[counter1][counter2]), 2)
            counter2 = counter2 + 1
        counter1 = counter1 + 1
    Ak = sum1
    Nr = len(AllRNA)
    rdpie = 1
    rd = rdpie / (Ak / Nr)
    
    # 生成RNAGaussian
    counter1 = 0
    while counter1 < len(AllRNA):   # 计算rna counter1和counter2之间的similarity
        counter2 = 0
        RNAGaussianRow = []
        while counter2 < len(AllRNA):   # 计算Ai*和Bj*
            AiMinusBj = 0
            sum2 = 0
            counter3 = 0
            AsimilarityB = 0
            while counter3 < len(AllDisease):   # rna的每个属性分量
                sum2 = pow((RNAAndDiseaseBinary[counter1][counter3] - RNAAndDiseaseBinary[counter2][counter3]), 2)#计算平方
                AiMinusBj = AiMinusBj + sum2
                counter3 = counter3 + 1
            AsimilarityB = math.exp(- (AiMinusBj*rd))
            RNAGaussianRow.append(AsimilarityB)
            counter2 = counter2 + 1
        RNAGaussian.append(RNAGaussianRow)
        counter1 = counter1 + 1
      
    # # RNASimilarity
    Func_Similarity = []
    counter = 0
    while counter < len(FuncSim):
        counter1 = 0
        Row = []
        while counter1 < len(FuncSim[counter]):
            if FuncSim[counter][counter1] == 0:
                Row.append(RNAGaussian[counter][counter1])
            else:
                v=(FuncSim[counter][counter1]+RNAGaussian[counter][counter1])/2
                Row.append(v)
            counter1 = counter1 + 1
        Func_Similarity.append(Row)
        counter = counter + 1  
    
    RNASimilarity = []
    counter=0
    while counter< len(FuncSim):
        FeaturePair = []
        FeaturePair.extend(Func_Similarity[counter])
        FeaturePair.extend(SeqSim[counter])
        FeaturePair.extend(Famsim[counter])
        RNASimilarity.append(FeaturePair) 
        counter=counter+1    
    RNASimilarity= Autoencoder(RNASimilarity)
    print("len(RNASimilarity[0]):",len(RNASimilarity[0]))
    
    DiseaseSimilarity = []
    counter = 0
    while counter < len(DiseaseSimilarityModel1):
        counter1 = 0
        Row = []
        while counter1 < len(DiseaseSimilarityModel1[counter]):
            v = (DiseaseSimilarityModel1[counter][counter1] + DiseaseSimilarityModel2[counter][counter1]) / 2
            if v > 0:
                Row.append(v)
            if v == 0:
                Row.append(DiseaseGaussian[counter][counter1])
            counter1 = counter1 + 1
        DiseaseSimilarity.append(Row)
        counter = counter + 1 
    DiseaseSimilarity= Autoencoder(DiseaseSimilarity)
    print("len(DiseaseSimilarity[0]):",len(DiseaseSimilarity[0]))  

     #训练样本特征
    TrainSampleFeature = []
    counter = 0
    while counter < len(X_train):     #遍历
        DN = X_train[counter][1]     # disease name
        RN = X_train[counter][0]     # rna name
        counter1 = 0
        while counter1 < len(AllDisease):
            if AllDisease[counter1][0] == DN:            
                counter2 = 0
                while counter2 < len(AllRNA):
                    if AllRNA[counter2][0] == RN:
                        FeaturePair = []  
                        FeaturePair.extend(RNASimilarity[counter2])
                        FeaturePair.extend(DiseaseSimilarity[counter1])
                        TrainSampleFeature.append(FeaturePair)                      
                        break
                    counter2 = counter2 + 1
                break
            counter1 = counter1 + 1
        counter = counter + 1    
    
    #测试样本特征
    TestSampleFeature = []
    counter = 0
    while counter < len(X_test):     #遍历
        DN = X_test[counter][1]     # disease name
        RN = X_test[counter][0]     # rna name
        counter1 = 0
        while counter1 < len(AllDisease):
            if AllDisease[counter1][0] == DN:            
                counter2 = 0
                while counter2 < len(AllRNA):
                    if AllRNA[counter2][0] == RN:
                        FeaturePair = []  
                        FeaturePair.extend(RNASimilarity[counter2])
                        FeaturePair.extend(DiseaseSimilarity[counter1])
                        TestSampleFeature.append(FeaturePair)                      
                        break
                    counter2 = counter2 + 1
                break
            counter1 = counter1 + 1
        counter = counter + 1
    TrainSampleFeature=np.array(TrainSampleFeature)  
    TestSampleFeature=np.array(TestSampleFeature)  
    return TrainSampleFeature,TestSampleFeature


def Autoencoder(SampleFeature):
    SampleFeature = np.array(SampleFeature)
    x = SampleFeature 
    x_train, x_test, y_train, y_test = train_test_split(x, x, test_size=0.2, random_state=32)    # 切分数据集进行训练，用全部数据集x进行“预测”

    # 改变数据类型
    x_train = x_train.astype('float32') / 1.
    x_test = x_test.astype('float32') / 1.

    # 变量
    encoding_dim = 128
    input_size = len(SampleFeature[0])

    # input
    input_img = Input(shape=(input_size,))    

    # encoding layer
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # decoding layer
    decoded = Dense(input_size, activation='sigmoid')(encoded)

    # build autoencoder, encoder
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)

    # compile autoencoder
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # training
    autoencoder.fit(x, x, epochs=300, batch_size=256, shuffle=True, validation_data=(x_test, x_test))   

    # 预测
    encoded_imgs = encoder.predict(x)
    encoded_imgs = np.array(encoded_imgs)        
    return encoded_imgs


def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)   #recall
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)    #precision
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F=2* Prec * Sen/ (Prec + Sen)
    # 分母可能出现0，需要讨论待续
    print('Accuracy:', Acc)
    print('Sen/recall:', Sen)
    print('Spec:', Spec)
    print('precision:', Prec)
    print('Mcc:', MCC)
    print('f1-score:', F)
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    Result.append(round(F, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    SumF=0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        SumF = SumF + matrix[counter][5]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    print('AverageF:', SumF / len(matrix))
    return

def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)   # 样本标准偏差
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制，保留两位小数
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result


def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter][1])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction

def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    # 第一个框的坐标，大小，第二个框的坐标，倍数，全部图像的fpr，tpr传入，粗细，颜色
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)  # 生成列的横坐标，横坐标都是x0，纵坐标变化
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return
    # 画虚线框
    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    # 连接两个虚线框
    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    # 小虚框内各点坐标
    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    # 画虚线框内的点
    # 把小虚框内的任一点减去小虚框左下角点生成相对坐标，再乘以倍数（4）加大虚框左下角点
    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, linestyle='--', color=color, lw=thickness, alpha=1)
    return


# 读取源文件
PositiveSample = []
ReadMyCsv(PositiveSample, "/home/Daiqiuyin/D3数据集/data/PositiveSample.csv")
NegativeSample = []
ReadMyCsv(NegativeSample, "/home/Daiqiuyin/D3数据集/data/NegativeSample.csv")
Sample=PositiveSample+NegativeSample

# SampleLabel
SampleLabel = []
counter = 0
while counter < len(Sample) / 2:
    # Row = []
    # Row.append(1)
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(Sample) / 2:
    # Row = []
    # Row.append(0)
    SampleLabel.append(0)
    counter1 = counter1 + 1

cv = RepeatedStratifiedKFold(n_splits=5,  n_repeats=5)

X = np.array(Sample)
y = np.array(SampleLabel)

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

from gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score
import pickle

# deep forest
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 384
    ca_config["max_layers"] = 10  #最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2   #判别的类别数量
    ca_config["estimators"] = []
    
    rf = {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400, "max_features": 4, 
          "min_samples_split": 12, "max_depth": None, "n_jobs": -1}
    
    rf_2 = {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500, "max_features": 6, 
          "min_samples_split": 10, "max_depth": None, "n_jobs": -1}

    xgb = {"n_folds": 5, "type": "XGBClassifier", 'booster':'gbtree', 'colsample_bylevel':1, 
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate': 0.01,
           'max_delta_step':0, 'max_depth':6, 'min_child_weight':0.001,
           'missing':None, 'n_estimators':600, 'n_jobs':-1, 'nthread':-1, 'objective':'binary:logistic', 
           'scale_pos_weight':1, 'seed':384, 'subsample':1}
    
    xgb_2 = {"n_folds": 5, "type": "XGBClassifier", 'booster':'gbtree', 'colsample_bylevel':1, 
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate': 0.1,
           'max_delta_step':0, 'max_depth':5, 'min_child_weight':0.1,
           'missing':None, 'n_estimators':200, 'n_jobs':-1, 'nthread':-1, 'objective':'binary:logistic', 
           'scale_pos_weight':1, 'seed':384, 'subsample':1}
    
    ca_config["estimators"].append(rf)
    ca_config["estimators"].append(rf_2) 
    ca_config["estimators"].append(xgb)
    ca_config["estimators"].append(xgb_2)     
    
    config["cascade"] = ca_config
    return config

aucs = []
AUPRs =[]
AllResult = []

# RandomForestClassifier ok!
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(max_features=4, min_samples_split=12, n_estimators=400, n_jobs=-1)
aucs2 = []
AUPRs2 =[]
AllResult2 = []


# LogisticRegression  ok!
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression(C=10, multi_class='multinomial',max_iter=1000, solver='newton-cg')
aucs3 = []
AUPRs3 =[]
AllResult3 = []

from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
aucs4 = []
AUPRs4 =[]
AllResult4 = []

from sklearn.svm import SVC  #非线性分类SVM
model5 = SVC(C=624, gamma=0.002,probability=True)
aucs5 = []
AUPRs5 =[]
AllResult5 = []

import xgboost
from xgboost import XGBClassifier
model6 = XGBClassifier( learning_rate=0.01, max_depth=6, min_child_weight=0.001, n_estimators=800, booster='gbtree',colsample_bylevel=1, colsample_bytree=1, eval_metric='auc', gamma=0,max_delta_step=0,nthread=-1,objective='binary:logistic', scale_pos_weight=1, seed=1231, subsample=1,n_jobs=-1)
aucs6 = []
AUPRs6 =[]
AllResult6 = []

i=0
for train, test in cv.split(X, y):
    print(i)
    X_train,X_test=FeatureGenerate(X[train],y[train], X[test])
    
    #deep forest
    config=get_toy_config()
    gc = GCForest(config)
    X_train_enc = gc.fit_transform(X_train, y[train])   
    y_pred = gc.predict(X_test)  #返回预测标签
    y_score1 = gc.predict_proba(X_test)  #返回预测属于某标签的概率
       
    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs.append(AUPR)
        
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    AllResult[i].append(AUPR)
       
    #保存Real,PredictionProb.csv
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/deepforest' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    # RandomForestClassifier
    model2.fit(X_train, y[train])
    y_score1 = model2.predict_proba(X_test)  #返回预测属于某标签的概率
    y_pred = model2.predict(X_test)#返回预测标签
    
    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs2.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs2.append(AUPR)
       
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult2.append(Result)
    AllResult2[i].append(roc_auc)
    AllResult2[i].append(AUPR)
       
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/rf' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    # LogisticRegression
    model3.fit(X_train, y[train])
    y_score1 = model3.predict_proba(X_test)  #返回预测属于某标签的概率
    y_pred = model3.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs3.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs3.append(AUPR)
        
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult3.append(Result)
    AllResult3[i].append(roc_auc)
    AllResult3[i].append(AUPR)
       
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/LogisticRegression' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    # NaiveBayes
    model4.fit(X_train, y[train])
    y_score1 = model4.predict_proba(X_test)  #返回预测属于某标签的概率
    y_pred = model4.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs4.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs4.append(AUPR)
    
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult4.append(Result)
    AllResult4[i].append(roc_auc)
    AllResult4[i].append(AUPR)
       
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/NaiveBayes' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    # SVM
    model5.fit(X_train, y[train])
    y_score1 = model5.predict_proba(X_test)  #返回预测属于某标签的概率
    y_pred = model5.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs5.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs5.append(AUPR)
        
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult5.append(Result)
    AllResult5[i].append(roc_auc)
    AllResult5[i].append(AUPR)
       
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/SVM' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    # xgboost
    model6.fit(X_train, y[train])
    y_score1 = model6.predict_proba(X_test)  #返回预测属于某标签的概率
    y_pred = model6.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    aucs6.append(roc_auc) 
    
    precision, recall,_ = precision_recall_curve(y[test], y_score1[:, 1])
    AUPR=auc(recall,precision)
    AUPRs6.append(AUPR)
    
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult6.append(Result)
    AllResult6[i].append(roc_auc)
    AllResult6[i].append(AUPR)
       
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3重复5折/result/xgb' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    

    i=i+1

Prediction_result = MyStd(AllResult)
StorFile(Prediction_result, '/home/Daiqiuyin/D3数据集/3重复5折/result/gc.csv')    

Prediction_result2 = MyStd(AllResult2)
StorFile(Prediction_result2, '/home/Daiqiuyin/D3数据集/3重复5折/result/rf.csv')    

Prediction_result3 = MyStd(AllResult3)
StorFile(Prediction_result3, '/home/Daiqiuyin/D3数据集/3重复5折/result/lr.csv')    

Prediction_result4 = MyStd(AllResult4)
StorFile(Prediction_result4, '/home/Daiqiuyin/D3数据集/3重复5折/result/nb.csv')    

Prediction_result5 = MyStd(AllResult5)
StorFile(Prediction_result5, '/home/Daiqiuyin/D3数据集/3重复5折/result/svm.csv')    

Prediction_result6 = MyStd(AllResult6)
StorFile(Prediction_result6, '/home/Daiqiuyin/D3数据集/3重复5折/result/xgb.csv')    