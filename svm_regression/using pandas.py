import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

path = r'/Users/mingzhao/PycharmProjects/MachineLearning/svm_regression/AM1L-1.csv'

data = pd.read_csv(path)
print(data.shape)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]

test = SelectKBest(score_func = chi2, k=4)
fit = test.fit[X, Y]

print(fit.scores_)
features = fit.transform(X)

print(features[0:5, :])

'''data = pd.read_csv(path)
print(data)

import matplotlib.pyplot as plt
data.hist()
plt.show()'''
