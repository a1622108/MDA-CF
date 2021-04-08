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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
from math import e,log
from numpy import interp
from itertools import cycle
from gcforest.gcforest import GCForest
import pickle
from sklearn.ensemble import RandomForestClassifier
import os           # https://blog.keras.io/building-autoencoders-in-keras.html
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 只显示WARNING（警告）、ERROR（错误）

import os           # https://blog.keras.io/building-autoencoders-in-keras.html
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 只显示WARNING（警告）、ERROR（错误）
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import csv
import math
import random
from keras import regularizers
from sklearn.model_selection import train_test_split

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

cv = StratifiedKFold(n_splits=5, random_state=32, shuffle=True)

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
from sklearn.ensemble import RandomForestClassifier

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

tprs = []
mean_fpr = np.linspace(0, 1, 100)   #横坐标

aucs = []
AUPRs =[]
AllResult = []
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']
i=0

for train, test in cv.split(X, y):
    X_train,X_test=FeatureGenerate(X[train],y[train], X[test])      
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
    
    #画图
    tprs.append(interp(mean_fpr, fpr, tpr))   # 插值函数interp,interp(mean_fpr, fpr, tpr)得到与mean_fpr对应的一系列y值(tpr)
    tprs[-1][0] = 0.0     # 初始处为0
    plt.plot(fpr, tpr, lw=1, alpha=0.3, color=colorlist[i],label='Fold %d (AUC = %0.4f)' % (i, roc_auc))
    MyEnlarge(0.05, 0.78, 0.2, 0.2, 0.35, 0.3, 2, mean_fpr, tprs[i], 1, colorlist[i]) 
    
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult.append(Result)
    AllResult[i].append(roc_auc)
    AllResult[i].append(AUPR)
       
    #保存Real,PredictionProb.csv
    RealAndPredictionProb = MyRealAndPredictionProb(y[test],y_score1)
    NameProb = '/home/Daiqiuyin/D3数据集/3每折/result/deepforest' + str(i) + 'Prob.csv'
    StorFile(RealAndPredictionProb, NameProb)    
    
    i=i+1

MyAverage(AllResult)
Prediction_result = MyStd(AllResult)
StorFile(Prediction_result, '/home/Daiqiuyin/D3数据集/3每折/gc.csv')    

# 画均值
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0   #list的最后一个值，即纵坐标最后一个点为（1,1）
mean_auc = np.mean(aucs, axis=0)    #计算平均AUC值
plt.plot(mean_fpr, mean_tpr, label='Mean (AUC = %0.4f)' % (mean_auc), linestyle='--', lw=2, alpha=.8, color=colorlist[5])

# 画虚线框
MyEnlarge(0.05, 0.78, 0.2, 0.2, 0.35, 0.3, 2, mean_fpr, mean_tpr, 2, colorlist[5])

# 画标题坐标轴
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
# plt.title('Receiver operating characteristic')
plt.legend(loc="lower right",fontsize=8)
plt.savefig('/home/Daiqiuyin/D3数据集/3每折/gc.png',dpi=300)

