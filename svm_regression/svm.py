import quandl


def get_finance_data(quandl_name):
    df = quandl.get(quandl_name)
    return df


def calculate_stats(df):
    df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    return df


def main():
    google_finance_data = get_finance_data("WIKI/GOOGL")
    print(google_finance_data.head())

    google_stats = calculate_stats(google_finance_data)
    print(google_stats.head())


if __name__ == '__main__':
    main()
