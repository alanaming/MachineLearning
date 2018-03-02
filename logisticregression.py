from sklearn import datasets
from sklearn import metrics
'''from sklearn.linear_model import LogisticRegression'''
'''from sklearn.naive_bayes import GaussianNB'''
'''#load iris for logisticregression
dataset = datasets.load_iris()

#create modle
logisticregression =LogisticRegression()
logisticregression.fit(dataset.data, dataset.target)

expected = dataset.target
predicted = logisticregression.predict(dataset.data)

#summary
print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted)'''

'''nb =GaussianNB()
nb.fit(dataset.data, dataset.target)'''
'''from sklearn.neighbors import KNeighborsClassifier
dataset = datasets.load_iris()
neighbor = KNeighborsClassifier()
neighbor.fit(dataset.data, dataset.target)

expected = dataset.target
predicted = neighbor.predict(dataset.data)'''

from sklearn.svm import SVC
dataset =datasets.load_iris()
SVM = SVC()
SVM.fit(dataset.data, dataset.target)
expected = dataset.target
predicted = SVM.predict(dataset.data)

print (metrics.classification_report(expected, predicted))
print (metrics.confusion_matrix(expected, predicted))


print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))