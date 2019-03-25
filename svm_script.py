# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import svm

#reading data. Posteriorly change to API service
df = pd.read_csv('...\\AAPL.csv')
df_index = pd.read_csv('...\\^GSPC.csv')

df_index_volume = list(df_index['Volume'])
df_index = list(df_index['Close'])
df_volume = list(df['Volume'])
df = list(df['Close'])

# alligning difference between X and Y. Removing the 1st element from feature array
#df_index.pop(0)
#df.pop(0)

dateParameter = 5

#MOMENTUM for stock
feature1 = []
for i in range(0, len(df)-dateParameter-1):
    feature1.append(1 if df[i+1]>df[i] else -1)
 
#MOMENTUM for index
feature2 = []
for i in range(0, len(df_index)-dateParameter-1):
    feature2.append(1 if df_index[i+1]>df_index[i] else -1)

#Growth rate for stock
feature3 = []
for i in range(0, len(df)-dateParameter-1):
    priceGrowthRate = 100*((df[i+1] - df[i])/df[i])
    feature3.append(priceGrowthRate)
   
#Growth rate for index
feature4 = []
for i in range(0, len(df_index)-dateParameter-1):
    priceGrowthRate_index = 100*((df_index[i+1] - df_index[i])/df_index[i])
    feature4.append(priceGrowthRate_index)
 
#VolumeChange for stock
feature5 = []
for i in range(0, len(df_volume)-dateParameter-1):
    volumeStockChange = 100*((df_volume[i+1] - df_volume[i])/df_volume[i])
    feature5.append(volumeStockChange)

#Volume Changerate for Index
feature6 = []
for i in range(0, len(df_index_volume)-dateParameter-1):
    volumeIndexChange = 100*((df_index_volume[i+1] - df_index_volume[i])/df_index_volume[i])
    feature6.append(volumeIndexChange)
   

X = np.transpose(np.array([feature1, feature2, feature3, feature4,feature5, feature6]))

#transforming our target Y to fit -1 or 1 if grew:
Y=[]
for i in range(dateParameter, len(df)-1):
    Y.append(1 if df[i+1] > df[i] else -1)

#Split data 70% 30%
split = int(251*0.7)

# X is our feature Vector with fatures arrays
train_X = np.array(X[0:split]).astype('float64') # Feature vector should be a 2D array. [[]]
test_X = np.array(X[split:]).astype('float64')  # Feature vector should be a 2D array. [[]]
train_Y = np.array(Y[0:split]).astype('float64')
test_Y = np.array(Y[split:]).astype('float64')

rbf_svm = svm.SVC(kernel='rbf')
rbf_svm.fit(train_X, train_Y)
score = rbf_svm.score(test_X, test_Y)
print(score)


