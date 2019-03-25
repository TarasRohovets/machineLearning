# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import svm

#Date paramenter 1 day, 1 week - 5, 1 month - 20, 3months - 60
# STOCKS - MSFT, AAPL, V, INTC, CSCO, MA, ORCL, ADBE, IBM, CRM

dateParameter = 60

#Momentum of STOCK and INDEX
def calcMomentum(df):
    momentumArray = []
    for i in range(0, len(df)-dateParameter-1):
        momentumArray.append(1 if df[i+1]>df[i] else -1)
    return momentumArray

#GrowthRate of STOCK and INDEX
def priceGrowthRate(df):
    growthRateArray = []
    for i in range(0, len(df)-dateParameter-1):
        priceGrowthRate = 100*((df[i+1] - df[i])/df[i])
        growthRateArray.append(priceGrowthRate)
    return growthRateArray

#VolumeChange for STOCK and INDEX
def calcVolumeChange(df):
    volumeChangeArray = []
    for i in range(0, len(df)-dateParameter-1):
        volumeIndexChange = 100*((df[i+1] - df[i])/df[i])
        volumeChangeArray.append(volumeIndexChange)
    return volumeChangeArray
    

def calculateFeatures(df, df_index, df_volume, df_index_volume):
    feature1 = calcMomentum(df)
    feature2 = calcMomentum(df_index)
    feature3 = priceGrowthRate(df)
    feature4 = priceGrowthRate(df_index)
    feature5 = calcVolumeChange(df_volume)
    feature6 = calcVolumeChange(df_index_volume)
    X = np.transpose(np.array([feature1 ,feature2, feature3 ,feature4, feature5, feature6]))
    return X
    
def creatingTargetArray(df):
    
    #transforming our target Y to fit -1 or 1 if grew:
    Y=[]
    for i in range(dateParameter, len(df)-1):
        Y.append(1 if df[i+1] > df[i] else -1)
    return Y

def svmMethod():
    #reading data. Posteriorly change to API service
    df = pd.read_csv('C:\\Users\\taras.rohovets\\Desktop\\Training Stuff\\Data\\CRM.csv')
    df_index = pd.read_csv('C:\\Users\\taras.rohovets\\Desktop\\Training Stuff\\Data\\^GSPC.csv')
    
    df_index_volume = list(df_index['Volume'])
    df_index = list(df_index['Close'])
    df_volume = list(df['Volume'])
    df = list(df['Close'])
   
    X = calculateFeatures(df, df_index, df_volume, df_index_volume)
    Y = creatingTargetArray(df)

    #Split data 70% 30%
    split = int(2263*0.7)
    
    # X is our feature Vector with fatures arrays
    train_X = np.array(X[0:split]).astype('float64') # Feature vector should be a 2D array. [[]]
    test_X = np.array(X[split:]).astype('float64')  # Feature vector should be a 2D array. [[]]
    train_Y = np.array(Y[0:split]).astype('float64')
    test_Y = np.array(Y[split:]).astype('float64')
    
    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm.fit(train_X, train_Y)
    score = rbf_svm.score(test_X, test_Y)
    print(score)

if __name__ == "__main__":
    svmMethod()