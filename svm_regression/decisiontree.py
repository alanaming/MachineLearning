from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#load the iris dataset
dataset = datasets.load_iris()
tree = DecisionTreeClassifier()

tree.fit(dataset.data, dataset.target)

expected = dataset.target
predicted = tree.predict(dataset.data)

#summarize the results

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))