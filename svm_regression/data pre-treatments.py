import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

path = r'/Users/mingzhao/PycharmProjects/MachineLearning/svm_regression/AM1L-1.csv'
dataframe = pd.read_csv(path)

print (dataframe)
array =dataframe.values


X=array[:, 0:101]
Y=array[1, :]

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

np.set_printoptions(precision=3)
print(rescaledX[0:5, :])

