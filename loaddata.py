# load csv

import csv
import numpy

path = r'/Users/mingzhao/PycharmProjects/MachineLearning/svm_regression/AM1L-1.csv'
data = open(path, 'rt')
reader = csv.reader(data, delimiter = ',', quoting = csv.QUOTE_NONE)
next(reader, None)
X = list(reader)

data = numpy.array(X).astype('float')
print(data)

#load with numpy
import numpy as np
path = r'/Users/mingzhao/PycharmProjects/MachineLearning/svm_regression/AM1L-1.csv'
data = open(path, 'rt')
load = np.loadtxt(data, delimiter=',', skiprows=1)
print(load.shape)