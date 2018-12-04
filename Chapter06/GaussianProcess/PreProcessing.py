import pandas as pd
import numpy as np


class PreProcessing:
    data = None
    quarter_names = None
    num_years = None
    num_days = None

    def __init__(self, name):
        name= str(name)
        self.get_data(name)
        self.data['Normalized_Close'] = self.normalized_data_col(self.data)
        self.data['Quarter'] = self.get_quarter_col(self.data)
        self.num_days = 252
        self.prices_by_year = self.get_prices_by_year()
        self.quarter_length = int(self.num_days / 4)

    def get_prices_by_year(self):
        df = self.modify_first_year_data()
        for i in range(1, len(self.num_years)):
            df = pd.concat([df, pd.DataFrame(self.get_year_data(year=self.num_years[i], normalized=True))], axis=1)

        df = df[:self.num_days]

        quarter_col = []
        num_days_in_quarter = self.num_days // 4
        for j in range(0, len(self.quarter_names)):
            quarter_col.extend([self.quarter_names[j]]*num_days_in_quarter)
        quarter_col = pd.DataFrame(quarter_col)

        df = pd.concat([df, quarter_col], axis=1)
        df.columns = self.num_years + ['Quarter']
        df.index.name = 'Day'

        df = self.fill_nans_with_mean(df)

        return df

    def get_year_data(self, year, normalized=True):
        year = int(year)
        if year not in self.num_years:
            raise ValueError('\n' +
                             'Input year: {} not in available years: {}'.format(year, self.num_years))

        prices = (self.data.loc[self.data['Date'].dt.year == year])
        if normalized:
            return np.asarray(prices.loc[:, 'Normalized_Close'])
        else:
            return np.asarray(prices.loc[:, 'Adj Close'])

    def get_adj_close_prices(self, start_year, end_year):
        start_year,end_year  = int(start_year), int(end_year)
        if start_year < self.num_years[0] or end_year > self.num_years[-1]:
            raise ValueError('\n' +
                             'Incorrect data! \n' +
                             'Max range available: {}-{}\n'.format(self.num_years[0], self.num_years[-1]) +
                             'Was: {}-{}'.format(start_year, end_year))

        df = (self.data.loc[(self.data['Date'].dt.year >= start_year) & (self.data['Date'].dt.year <= end_year)])
        df = df.loc[:, ['Date', 'Adj Close']]

        return df

    def get_data(self, file_name):
        file_name = str(file_name)
        self.data = pd.read_csv('Data/' + file_name + '.csv')
        self.data = self.data.iloc[:, [0, 5]]
        self.data = self.data.dropna()
        self.data.Date = pd.to_datetime(self.data.Date)
        self.quarter_names = ['Q' + str(i) for i in range(1, 5)]

    def normalized_data_col(self, df):
        price_normalized = pd.DataFrame()

        date_list = list(df.Date)
        self.num_years = sorted(list(set([date_list[i].year for i in range(0, len(date_list))])))

        for i in range(0, len(self.num_years)):
            prices_data = self.get_year_data(year=self.num_years[i], normalized=False)
            prices_data = [(prices_data[i] - np.mean(prices_data)) / np.std(prices_data) for i in range(0, len(prices_data))]
            prices_data = [(prices_data[i] - prices_data[0]) for i in range(0, len(prices_data))]
            price_normalized = price_normalized.append(prices_data, ignore_index=True)

        return price_normalized

    def get_quarter_col(self, df):
        quarters = pd.DataFrame()

        for i in range(0, len(self.num_years)):
            dates = list((df.loc[df['Date'].dt.year == self.num_years[i]]).iloc[:, 0])
            dates = pd.DataFrame([self.quarter_names[(int(dates[i].month) - 1) // 3] for i in range(0, len(dates))])
            quarters = quarters.append(dates, ignore_index=True)

        return quarters


    def modify_first_year_data(self):
        price_data = pd.DataFrame(self.get_year_data(self.num_years[0]))
        df = pd.DataFrame([0 for _ in range(self.num_days - len(price_data.index))])
        df = pd.concat([df, price_data], ignore_index=True)

        return df

    def fill_nans_with_mean(self, df):
        years = self.num_years[:-1]
        df_wo_last_year = df.loc[:,years]
        df_wo_last_year = df_wo_last_year.fillna(df_wo_last_year.mean())
        df_wo_last_year[self.num_years[-1]] = df[self.num_years[-1]]
        df= df_wo_last_year

        return df
