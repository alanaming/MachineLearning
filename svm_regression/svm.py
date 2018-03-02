
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm


def get_digits_data():
    return datasets.load_digits()


def smv_classification(digits):
    clf = svm.SVC(gamma=0.001, C=100)
    x, y = digits.data[:-1], digits.target[:-1]
    clf.fit(x, y)
    plt.imshow(digits.images[-1], 'spring')

    print('prediction', clf.predict(digits.data[-1]))


def main():
    digits = get_digits_data()
    print(digits)


    digits = get_digits_data()
    smv_classification(digits)


if __name__ == '__main__':
    main()
