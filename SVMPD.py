# -*- coding: UTF-8 -*-
from sklearn import svm,datasets
import pandas as pd
import scipy.io as sc
import numpy as np
import time
from scipy import stats
import operator#operator 函数来加快速度,
from sklearn import datasets,neighbors
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split
import sklearn.cross_validation as cross_validation
#sklearn.cross_validation
#KFold：K-Fold交叉验证迭代器。接收元素个数、fold数、是否清洗
#LeaveOneOut：LeaveOneOut交叉验证迭代器
#LeavePOut：LeavePOut交叉验证迭代器
#LeaveOneLableOut：LeaveOneLableOut交叉验证迭代器
#LeavePLabelOut：LeavePLabelOut交叉验证迭代器
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import  BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB#高斯分布的朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB#多项式分布的朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB#伯努利分布的朴素贝叶斯
#from sklearn.ensemble import RandomTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from scipy.stats import uniform as sp_rand
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
#from sklearn.cluster import Ward
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import mixture
from sklearn import metrics

def readcsv(path):
    df=pd.read_csv(path)
    acc=np.array(df)
    label=acc[:,3]
    feature=acc[:,0:2]
    return acc

path= u'F:\学习\Parkinson\FOG\dataset\PD1ACCLabel.csv'
f = open(path)
feature = readcsv(f)
all = feature[1:]

#  select the single person
#idx = np.where(all[:, -1]==1)
#all = all[idx]
#print (all.shape)

# select the first 4th columns, dropout the person ID.
all = all[:, 0:4]
n_classes =2

 
# Cliping
print ('all', all.shape)
len_sample = 100
data_size_or=all.shape[0]
# clip the data, make sure it can be divided into four parts: 3parts for training and 1 part for testing
all = all[:4*len_sample*int(data_size_or/(4*len_sample))]
data_size = all.shape[0]
print ('all', all.shape)

no_fea= all.shape[1] - 1
F_ = all[:, 0:no_fea]
L_ = all[:, no_fea:no_fea+1]


##segmentation
## Sliding window
len_seg = 100
overlap = 50
_overlap = 100 - overlap  # the non-overlap part
seg = F_[0:len_seg]
print (seg.shape)
seg = seg[ np.newaxis,:]
print (seg.shape)
label_seg = np.transpose(L_[0:len_seg])  # the label vector of this segment
print ('label', label_seg.shape)

for i in range(1, int(data_size_or/_overlap-5)):
    new = F_[_overlap * i:_overlap * i + len_seg]
    new = new[np.newaxis, :]
    label_new = np.transpose(L_[_overlap * i:_overlap * i + len_seg])
    modes, _ = stats.mode(label_new, axis=1)
    # if the mean = modes, are the samples in this segment are from the same label, stack it.
    if np.mean(label_new) == modes:
        seg = np.vstack((seg, new))
        label_seg = np.vstack((label_seg, label_new))

## stacked the last segment doublely, make the datasize even
label_seg = label_seg[:,0:1]
print (seg.shape, label_seg.shape, sum(label_seg))

#zip
zipped = zip(seg, label_seg)
#np.random.shuffle(list(zipped))
seg, label_seg = zip(*zipped)
seg = np.array(seg)
label_seg = np.array(label_seg)


data_size = seg.shape[0]
seg = seg[:4*int(data_size/4)]
label_seg = label_seg[:4*int(data_size/4)]

data_size = seg.shape[0]
middle = int(data_size*0.75)

feature_training = seg[0: middle]
label_training =label_seg[0:middle]
feature_testing =seg[middle: data_size]
label_testing =label_seg[middle: data_size]

print(feature_testing.shape,label_testing.shape, feature_training.shape)



def Acquire_Data_ALL_List(path_input):
                #start = time.clock()
                f=open(path_input)
                df=pd.read_csv(f)[1:]
                #print list(df)
                #print df
                TotalTime=list(df)[0]
                dataline=list(df)[1]
                #print 'TotalTime=',TotalTime,',dataline=',dataline
                group_XYZW=np.array(df)
                #print group_XYZW
                #labels=(np.array([group_XYZW[:,3]])).T
                labels=group_XYZW[:,3]
                labels=list(labels)
                #print labels[0],type(labels[0]),type(labels),len(labels)
                group_XYZ=(np.array([group_XYZW[:,0],group_XYZW[:,1],group_XYZW[:,2]])).T
                group_XY=(np.array([group_XYZ[:,0],group_XYZ[:,1]])).T
                group_XZ=(np.array([group_XYZ[:,0],group_XYZ[:,2]])).T
                group_YZ=(np.array([group_XYZ[:,1],group_XYZ[:,2]])).T
                group_X=group_XYZW[:,0]
                
                group_Y=group_XYZW[:,1]
                group_Z=group_XYZW[:,2]
                group_combined=(group_X**2+group_Y**2+group_Z**2)**0.5
                group_X=list(group_X)
                group_Y=list(group_Y)
                group_Z=list(group_Z)
                group_combined=list(group_combined)
                print (u'验证数据是否正确',type(labels)==type(group_X))
                #print group_combined,group_combined.shape
                #print group_X[0][0], group_Y[0][0], group_Z[0][0],group_X[len(group_XYZ)-1][0],group_Y[len(group_XYZ)-1][0],group_Z[len(group_XYZ)-1][0],len(group_XYZ)
                #print group_XYZ,group_XY,group_XZ,group_X,group_combined,group_YZ,group_Y,group_Z
                #end = time.clock()
                #print "csv_pandas的执行%f秒" % (end - start)
                #print group_X[0]
                return labels,group_XYZ,group_XY,group_XZ,group_X,group_combined,group_YZ,group_Y,group_Z

def nextpow2(i):
    """
    @brief Find 2^n that is equal to or greater than.
    @param i [int] 数据长度
    @return 返回i长度的下个2**n次的n
    """
    n = 1
    while n < i: n *= 2
    return n
            
def spectrum(datas,fs,fftSize = -1,fftType=1,scale='amp',isDetrend = True):
    """频谱分析
    Args:
    datas: 原始波形
    fs: 采样率
    fftSize:int 
        fft的长度，0 or -1 or int size -1为自动延拓，0为不指定长度由自身长度决定
    fftType:int
        fft类型，0 or 1 0为fft，1为rfft
    isDetrend:bool
        是否对信号进行去均值处理
    scale:string
        幅值处理方式：amp幅值Amplitude,ampDB为幅值加上分贝,mag为幅度谱，只是对fft结果取模，
    Return:
        (freqs,y):(nump.array,nump.array) 频率序列，幅值序列
    """
    if isDetrend:
        datas = datas - np.mean(datas)

    if 0 == fftSize:
        fftSize = len(datas)
    elif fftSize<0:
        fftSize = nextpow2(len(datas))


    if 0 == fftType:
        y = np.fft.fft(datas,fftSize)
        y = y[0:len(y)/2]
    else:
        y = np.fft.rfft(datas,fftSize)

    amp = dealFFTMagnitude(y,scale,fftSize)
    freqs = dealFFTFrequency(fs,fftSize)
    return (freqs,amp)

def dealFFTMagnitude(mag,scale='amp',fftSize=0):
    """fft之后的幅值处理
    Args:
    mag: fft之后的波形
    scale:string
        幅值处理方式：amp幅值Amplitude,ampDB为幅值加上分贝,mag为幅度谱，只是对fft结果取模，
    Return:
    """
    
    if 0 == fftSize:
        fftSize = len(mag)
    elif fftSize<0:
        fftSize = nextpow2(len(mag))
    temp = 2/fftSize;
    spectrumSize = int(fftSize/2);
    endIndex = spectrumSize - 1;
    amp = np.zeros((spectrumSize,len(mag[0])))
    scale = scale.lower()
    if scale == 'amp':
        amp[0] = np.abs(mag[0])/fftSize
        amp[endIndex] = np.abs(mag[endIndex])/fftSize
        amp[1:endIndex] = np.abs(mag[1:endIndex])*temp
    elif scale == 'ampdb':
        amp[0] = np.abs(mag[0])/fftSize
        amp[endIndex] = np.abs(mag[endIndex])/fftSize
        amp[1:endIndex] = np.abs(mag[1:endIndex])*temp
        amp = 20*np.log10(np.clip(np.abs(amp), 1e-20, 1e100))
    elif scale == 'mag':
        amp = np.abs(mag[0:endIndex+1])
    else:
        raise ValueError('scale=\'amp\' or \'ampdb\' or \'mag\' ')
    return amp

def dealFFTFrequency(fs,fftSize):
    """根据采样率和fft的数目计算频率分布
    Args:
        fs: 采样率
        fftSize:傅里叶变换长度
    Return:
        计算得到的频率分布np.array like
    """
    freqs = np.linspace(0,fftSize/2-1,fftSize/2)*(fs/fftSize)
    return freqs
            
def sklearnSVM(k1,k2,k3,filename,i):
        [y, prob_XYZ,prob_XY,prob_XZ,prob_X,prob_combined,prob_YZ,prob_Y,prob_Z] =Acquire_Data_ALL_List(filename)
        #print u'testtest',type(prob_X),type(y)
        prob_XX=[prob_XYZ,prob_XY,prob_XZ,prob_X,prob_combined,prob_YZ,prob_Y,prob_Z][i]
        rates=[]
        for i in range(0,len(prob_XX)-k2+1,k1):
            #print i,len(prob_Y)-k2,int((len(prob_Y)-k2)/k1)*k1
            a=prob_XX[i:i+k2]
            freq,ass=spectrum(a,100,fftSize = -1,fftType=1,scale='amp',isDetrend = True)
            summ=0
            summxiao=0
            for j in range(len(freq)):
                if freq[j]>=3 and freq[j]<=8:
                    summ+=ass[j]
                if freq[j]>=0.5 and freq[j]<=3:
                    summxiao+=ass[j]
            #print summxiao/summ,i,i+k1
            #rate=summ/ass.sum()
            rate=(summxiao/summ)**1
            rates.append(rate)
        #print rates,len(rates),int(len(prob_XX)/k1)-len(rates)
        tem=sum(rates[-4:])/len(rates[-4:])
        for i in range(int(len(prob_XX)/k1)-len(rates)):
                rates.append(tem)
        print (u'验证数据是否正确',int(len(prob_XX)/k1)==len(rates))
        #print rates
        yrates=[]
        for i in range(len(rates)):
                for j in range(k1):
                        yrates.append(rates[i])
        tem=yrates[-1]
        for i in range(int(len(prob_XX))-len(yrates)):
                yrates.append(tem)
        
        print (u'验证数据是否正确',int(len(prob_XX))==len(yrates))
        #print yrates[0],yrates[-1]
        
        totalNum = len(yrates)
        # 选出70%样本作为训练样本，其余30%测试
        trainNum = int(0.7 * totalNum)
        yrates=np.array([yrates]).T
        #yrates=np.reshape(yrates,(-1,totalNum)).T
        trainX = yrates[0 : trainNum]
        trainY = y[0 : trainNum]
        testX = yrates[trainNum:]
        testY = y[trainNum:]
        #trainX ,testX, trainY, testY = train_test_split(yrates, y, test_size = 0.3)
    #    titles = ['LinearSVC (linear kernel)',  
    #              'SVC with polynomial (degree 3) kernel',  
      #            'SVC with RBF kernel',  
      #            'SVC with Sigmoid kernel']  
        clf_linear  = svm.SVC(kernel='linear').fit(trainX, trainY)  
        #clf_linear  = svm.LinearSVC().fit(trainX, trainY)  
        clf_poly    = svm.SVC(kernel='poly', degree=3).fit(trainX, trainY)  
        clf_rbf     = svm.SVC().fit(trainX, trainY)  
        clf_sigmoid = svm.SVC(kernel='sigmoid').fit(trainX, trainY)
        #for clf in enumerate((clf_rbf)):
        #for clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
        SVMnames=[u'线性核函数',u'多项式核函数',u'RBF径向基核函数',u'sigmoid二层神经网络核函数']
        j=0
        for clf in [clf_linear, clf_poly, clf_rbf, clf_sigmoid]:
                #clf.fit(trainX, trainY)
                #print 'iiiiiii'
                Z = clf.predict(testX)
                #print len(Z),type(Z)
                TP=0
                TN=0
                FP=0
                FN=0
                pv=Z
                ty=testY
                for i in range(len(testY)):
                        if (pv[i]==1 and ty[i]==1):
                                TP=TP+1
                        if (pv[i]==-1 and ty[i]==-1):
                                TN=TN+1
                        if (pv[i]==1 and ty[i]==-1):
                                FP=FP+1
                        if (pv[i]==-1 and ty[i]==1):
                                FN=FN+1
                try:
                        print (u'%sSVM按时间评估：'%(SVMnames[j]))
                        j+=1
                        print ('Accuracy=',(TP+TN)/(TP+TN+FP+FN))
                        print ('Precision=',(TP)/(TP+FP))
                        #print 'Recall=',(TP)/(TP+FN)
                        print ('Sensitivity=',(TP)/(TP+FN))
                        print ('Speciality=',(TN)/(TN+FP))
                        #print ty.count(1)==TP+FN
                except(ZeroDivisionError):
                        print (u'分母为零，数据出错')
                
def getDatas(k1,k2,filename,i):
        #print filename
        [y, prob_XYZ,prob_XY,prob_XZ,prob_X,prob_combined,prob_YZ,prob_Y,prob_Z] =Acquire_Data_ALL_List(filename)#Acquire_Data_Label1_List
        #print u'testtest',type(prob_X),type(y)
        prob_XX=[y, prob_XYZ,prob_XY,prob_XZ,prob_X,prob_combined,prob_YZ,prob_Y,prob_Z][i]
        rates=[]
        for i in range(0,len(prob_XX)-k2+1,k1):
            #print i,len(prob_Y)-k2,int((len(prob_Y)-k2)/k1)*k1
            a=prob_XX[i:i+k2]
            freq,ass=spectrum(a,100,fftSize = -1,fftType=1,scale='amp',isDetrend = True)
            summ=0
            summxiao=0
            for j in range(len(freq)):
                if freq[j]>=3 and freq[j]<=8:
                    summ+=ass[j]
                if freq[j]>=0.5 and freq[j]<=3:
                    summxiao+=ass[j]
            #print summxiao/summ,i,i+k1
            #rate=summ/ass.sum()
            rate=(summxiao/summ)**1
            rates.append(rate)
        #print rates,len(rates),int(len(prob_XX)/k1)-len(rates)
        tem=sum(rates[-4:])/len(rates[-4:])
        for i in range(int(len(prob_XX)/k1)-len(rates)):
                rates.append(tem)
        print (u'验证数据是否正确',int(len(prob_XX)/k1)==len(rates))
        #print rates
        yrates=[]
        for i in range(len(rates)):
                for j in range(k1):
                        yrates.append(rates[i])
        tem=yrates[-1]
        for i in range(int(len(prob_XX))-len(yrates)):
                yrates.append(tem)
        
        print (u'验证数据是否正确',int(len(prob_XX))==len(yrates))
        #print yrates[0],yrates[-1]
        return yrates,y      

def Algorithms(k1,k2,filename,i):
        yrates,y=getDatas(k1,k2,filename,i)
        totalNum = len(yrates)
        # 选出70%样本作为训练样本，其余30%测试
        trainNum = int(0.7 * totalNum)
        yrates=np.array([yrates]).T
        yrates=np.reshape(yrates,(-1,totalNum)).T
        y=np.array(y).T
        #yratess=preprocessing.scale(yrates)
        #yrates=preprocessing.normalize(yrates)
        #yrates=preprocessing.scale(yrates)
        trainX = yrates[0 : trainNum]
        trainY = y[0 : trainNum]
        testX = yrates[trainNum:]
        testY = y[trainNum:]
        trainX ,testX, trainY, testY = train_test_split(yrates, y, test_size = 0.3)#不是K-Fold
        trainXLength=len(trainX)
        trainYLength=len(trainY)
        testXLength=len(testX)
        testYLength=len(testY)
        #print trainXLength,trainYLength,testXLength,testYLength
        '''
        AlgorithmNames=['KNN','linearSVM','polySVM','RBFkernelSVM','SigmoidkernelSVM','LR','LDA','RF','NBC','DTC','GBDT','AdaBoost',\
                        'GNB','BNB','LRCV','BGC','ETC','QD','Vote']
        Algorithm=[KNN(trainX,testX,trainY),linearSVM(trainX,testX,trainY),polySVM(trainX,testX,trainY),\
                   RBFkernelSVM(trainX,testX,trainY),SigmoidkernelSVM(trainX,testX,trainY),LR(trainX,testX,trainY),\
                   LDA(trainX,testX,trainY),RF(trainX,testX,trainY),NBC(trainX,testX,trainY),\
                   DTC(trainX,testX,trainY),GBDT(trainX,testX,trainY),AdaBoost(trainX,testX,trainY),\
                   GNB(trainX,testX,trainY),BNB(trainX,testX,trainY),\
                   LRCV(trainX,testX,trainY),BGC(trainX,testX,trainY),ETC(trainX,testX,trainY),\
                   QD(trainX,testX,trainY),Vote(trainX,testX,trainY)]
                   '''
        AlgorithmNames=['linearSVM','polySVM','RBFkernelSVM','SigmoidkernelSVM','LR','LDA','GBDT','AdaBoost',\
                        'GNB','LRCV','QD','Vote']
        AlgorithmNames=['linearSVM','polySVM','RBFkernelSVM','SigmoidkernelSVM','LR','LDA','GBDT','AdaBoost',\
                        'GNB','LRCV','QD','Vote','KNN','RF','NBC','DTC','BNB','BGC','ETC']
        '''
        Algorithm=[linearSVM(trainX,testX,trainY),polySVM(trainX,testX,trainY),\
                   RBFkernelSVM(trainX,testX,trainY),SigmoidkernelSVM(trainX,testX,trainY),LR(trainX,testX,trainY),\
                   LDA(trainX,testX,trainY),\
                   GBDT(trainX,testX,trainY),AdaBoost(trainX,testX,trainY),\
                   GNB(trainX,testX,trainY),\
                   LRCV(trainX,testX,trainY),\
                   QD(trainX,testX,trainY),Vote(trainX,testX,trainY)]
        '''
        #Algorithm=[RandomizedSearch(trainX,testX,trainY)]
        #print u'验证数据是否正确',len(AlgorithmNames)==len(Algorithm)
        Accuracys=[]
        Precisions=[]
        Sensitivitys=[]
        Specialitys=[]
        for i in range(1):
                '''
                #print i+1
                Z=Algorithm[i]
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,AlgorithmNames[i])
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                '''
                #1
                Z=linearSVM(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'linearSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #2
                Z=polySVM(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'polySVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #3
                Z=RBFkernelSVM(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'RBFkernelSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #4
                Z=SigmoidkernelSVM(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'SigmoidkernelSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #5
                Z=LR(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'LR')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #6
                Z=LDA(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'LDA')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #7
                Z=GBDT(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'GBDT')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #8
                Z=AdaBoost(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'AdaBoost')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #9
                Z=GNB(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'GNB')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #10
                Z=LRCV(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'LRCV')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #11
                Z=QD(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'QD')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #12
                Z=Vote(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'Vote')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)

                #13
                Z=KNN(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'linearSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #14
                Z=RF(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'polySVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #15
                Z=NBC(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'RBFkernelSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #16
                Z=DTC(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'SigmoidkernelSVM')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #17
                Z=BNB(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'LR')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #18
                Z=BGC(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'LDA')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
                #19
                Z=ETC(trainX,testX,trainY)
                Accuracy,Precision,Sensitivity,Speciality=testDatas(Z,testY,'GBDT')
                Accuracys.append(Accuracy)
                Precisions.append(Precision)
                Sensitivitys.append(Sensitivity)
                Specialitys.append(Speciality)
       # print Accuracys,Precisions,Sensitivitys,Specialitys       
        return Accuracys,Precisions,Sensitivitys,Specialitys,AlgorithmNames,trainXLength,testXLength   
def testDatas(Z,testY,name):
        TP=0
        TN=0
        FP=0
        FN=0
        pv=Z
        ty=testY
        Accuracy=0
        Precision=0
        Sensitivity=0
        Speciality=0
        for i in range(len(testY)):
                if (pv[i]==1 and ty[i]==1):
                        TP=TP+1
                if (pv[i]==-1 and ty[i]==-1):
                        TN=TN+1
                if (pv[i]==1 and ty[i]==-1):
                        FP=FP+1
                if (pv[i]==-1 and ty[i]==1):
                        FN=FN+1
        try:
                #print u'%s按时间评估：'%name
                #print 'Accuracy=',(TP+TN)/(TP+TN+FP+FN)
                #print 'Precision=',(TP)/(TP+FP)
                #print 'Recall=',(TP)/(TP+FN)
                #print 'Sensitivity=',(TP)/(TP+FN)
                #print 'Speciality=',(TN)/(TN+FP)
                #print ty.count(1)==TP+FN
                Accuracy=(TP+TN)/(TP+TN+FP+FN)
                Precision=(TP)/(TP+FP)
                Sensitivity=(TP)/(TP+FN)
                Speciality=(TN)/(TN+FP)
        except(ZeroDivisionError):
                print (u'分母为零，数据出错')
        #print TP,TN,FP,FN,TP/(((TP+FP)*(TP+FN))**0.5),TN/(((TN+FP)*(TN+FN))**0.5)
        return Accuracy,Precision,Sensitivity,Speciality      