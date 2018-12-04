import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from PreProcessing import PreProcessing
from GP import GP


class PlotData:
    company = None
    preprocessed_data = None
    gp_model = None

    def __init__(self, company):
        self.company = str(company)
        self.preprocessed_data = PreProcessing(str(company))
        self.gp_model = GP(str(company))

    def plot_normalized_prices(self, first_year, last_year):
        first_year, last_year = int(first_year), int(last_year)
        self.check_data(start_year=first_year, end_year=last_year)

        fig = plt.figure(num=self.company + ' normalized prices')
        ax = plt.gca()
        fig.set_size_inches(12, 6)
        lower_y, upper_y = 0, 0
        for year in range(first_year, last_year + 1):
            target = self.preprocessed_data.prices_by_year[year]
            lower_y = min(lower_y, min(target))
            upper_y = max(upper_y, max(target))
            x = np.linspace(0, len(target), len(target))
            plt.plot(x, target, alpha=.8, label=year)
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        y_max = max(abs(lower_y) - 1, abs(upper_y) + 1)
        x_min, x_max = -10, self.preprocessed_data.num_days + 10
        ax.set_ylim(bottom=-y_max, top=y_max)
        ax.set_xlim(left=x_min, right=x_max)

        for i in range(0, 5):
            plt.vlines(x=(self.preprocessed_data.num_days / 4) * i, ymin=-y_max, ymax=y_max, color='black', linestyles='--', alpha=.5,
                       zorder=-1)
            if i < 4:
                ax.text((self.preprocessed_data.num_days / 4) * i + self.preprocessed_data.num_days / 8 - 5, y_max - 0.5, self.preprocessed_data.quarter_names[i],
                        fontsize=12)
        plt.hlines(y=0, xmin=x_min, xmax=x_max, color='black', linestyles='--', alpha=.6, zorder=-1)

        plt.grid(True, alpha=.25)
        plt.title(self.company)
        plt.xlabel('Days')
        plt.ylabel('NormalizedPrices')

        plt.tight_layout()

        image_name = '{}_{}_{}_prices_normalized.png'.format(self.company, first_year, last_year)
        fig.savefig(image_name, dpi=fig.dpi)
        plt.clf()

    def plot_gp_predictions(self, train_start, train_end, pred_year, pred_quarters = None):
        train_start = int(train_start)
        train_end = int(train_end)
        pred_year = int(pred_year)
        self.check_data(start_year=train_start, end_year=pred_year)

        price_data = self.preprocessed_data.prices_by_year[pred_year]
        price_data = price_data[price_data.iloc[:].notnull()]

        fig = plt.figure(num=self.company + ' predicted prices')
        ax = plt.gca()
        fig.set_size_inches(12, 6)

        x_obs = list(range(price_data.index[0], price_data.index[-1] + 1))
        x_mesh, y_mean, y_var = self.gp_model.make_gp_predictions(start_year=train_start, end_year=train_end,
                                                                  pred_year=pred_year,
                                                                  pred_quarters=pred_quarters)
        y_lower = np.squeeze(y_mean - 1.96*np.sqrt(y_var))
        y_upper = np.squeeze(y_mean + 1.96*np.sqrt(y_var))
        y_max = max(abs(min(y_lower) - 1), abs(max(y_upper) + 1))
        ax.set_ylim(bottom=-y_max, top=y_max)

        x_min, x_max = -10, self.preprocessed_data.num_days + 10
        ax.set_xlim(left=x_min, right=x_max)

        plt.plot(x_obs, price_data, color='blue', alpha=.95, label=u'Actuals ' + str(pred_year), zorder=10)
        plt.plot(x_mesh, y_mean, color='red', linestyle='--', label=u'Predicted')
        plt.fill_between(x_mesh, y_lower, y_upper,
                         alpha=.25, label='95% confidence interval', color='red')

        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels, new_handles = [], []
        for handle, label in zip(handles, labels):
            if label not in new_labels:
                new_labels.append(label)
                new_handles.append(handle)
        plt.legend(new_handles, new_labels, bbox_to_anchor=(0.01, 0.02), loc='lower left', borderaxespad=0.)

        for i in range(0, 5):
            plt.vlines(x=self.preprocessed_data.quarter_length * i, ymin=-y_max, ymax=y_max, color='black', linestyles='--', alpha=.5,
                       zorder=-1)
            if i < 4:
                ax.text(self.preprocessed_data.quarter_length * i + self.preprocessed_data.quarter_length / 2 - 5, y_max - 0.5, self.preprocessed_data.quarter_names[i],
                        fontsize=12)
        plt.hlines(y=0, xmin=x_min, xmax=x_max, color='black', linestyles='--', alpha=.6, zorder=-1)

        plt.grid(True, alpha=.25)
        plt.title(self.company)
        plt.xlabel('Days\n')
        plt.ylabel('NormalizedPrices')

        plt.tight_layout()

        image_name = '{}_{}_predicted.png'.format(self.company, pred_year)
        fig.savefig(image_name, dpi=fig.dpi)
        plt.clf()

    def plot_complete_history(self, intermediate = False):
        self.plot_prices_data(start_year=self.preprocessed_data.num_years[0], end_year=self.preprocessed_data.num_years[-1], intermediate=intermediate)

    def plot_prices_data(self, start_year, end_year, intermediate = True):
        start_year,end_year = int(start_year), int(end_year)
        self.check_data(start_year=start_year, end_year=end_year)

        data = self.preprocessed_data.get_adj_close_prices(start_year=start_year, end_year=end_year)

        fig = plt.figure(num=self.company + ' prices')
        fig.set_size_inches(12, 6)
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], color='green', alpha=.95,
                 label=u'PriceData ' + str(start_year) + '-' + str(end_year), zorder=10)
        ax = plt.gca()

        x_ticks = [data[data['Date'].dt.year == year].iloc[0, 0] for year in range(start_year,end_year + 1)]
        x_ticks.append(data[data['Date'].dt.year == end_year].iloc[-1, 0]) # Appending the last date

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        if not intermediate:
            x_ticks = [x_ticks[0], x_ticks[-2], x_ticks[-1]]
            ax.set_xticks([x_ticks[0], x_ticks[-1]])
        else:
            ax.set_xticks(x_ticks)
        plt.xticks(rotation=20)
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_xlim(left=x_min, right=x_max)

        for i in range(0, len(x_ticks)):
            plt.vlines(x=x_ticks[i], ymin=y_min, ymax=y_max, color='black', linestyles='--', alpha=.5,
                       zorder=-1)

        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.title(self.company)
        plt.ylabel('Price')

        plt.tight_layout()

        fname = '{}_{}_{}_adj_closing_prices.png'.format(self.company, start_year, end_year)
        fig.savefig(fname, dpi=fig.dpi)
        plt.clf()

    def check_data(self, start_year, end_year):
        if int(start_year) < self.preprocessed_data.num_years[0] or int(end_year) > self.preprocessed_data.num_years[-1]:
            raise ValueError('\n' +
                             'Incorrect data! \n' +
                             'Max range available: {}-{}\n'.format(self.preprocessed_data.num_years[0], self.preprocessed_data.num_years[-1]) +
                             'Was: {}-{}'.format(int(start_year), int(end_year)))
