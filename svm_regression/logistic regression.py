#logistic regression
from sklearn import datasets
from sklÅearn import metrics
'''from sklearn.linear_model import LogisticRegression

from sklearn .naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier'''

from sklearn.svm import SVC

dataset = datasets.load_iris()

''''#create a model
logisticregression = LogisticRegression()
logisticregression.fit(dataset.data, dataset.target'''

'''nb = GaussianNB()
nb.fit(dataset.data, dataset.target)

neighbor = KNeighborsClassifier()
neighbor.fit(dataset.data, dataset.target)'''

SVM = SVC()
SVM.fit(dataset.data, dataset.target)

expected = dataset.target
'''predicted = logisticregression.predict(dataset.data)'''
'''predicted = neighbor.predict(dataset.data)'''
predicted =SVM.predict(dataset.data)

#summary
print (metrics.classification_report(expected, predicted))
print (metrics.confusion_matrix(expected, predicted))

