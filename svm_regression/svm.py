import quandl
import statistics
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
    print(digits)
    print('prediction', clf.predict(digits.data[-1]))


def get_finance_data(quandl_name):
    return quandl.get(quandl_name)


def calculate_stats(df):
    df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    return df


def calculate_mean(df):
    mean1 = statistics.mean(df['Adj. Close'])
    return mean1


def main():
    google_finance_data = get_finance_data("WIKI/GOOGL")
    print(google_finance_data.head())

    google_stats = calculate_stats(google_finance_data)
    print(google_stats.head())

    mean_statistics = calculate_mean(google_finance_data)
    print(str(mean_statistics))

    digits = get_digits_data()
    smv_classification(digits)


if __name__ == '__main__':
    main()
