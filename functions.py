
from nsepy import get_history
import pprint
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import requests
import tqdm
import sys
from matplotlib.backends.backend_pdf import PdfPages
import time

### Helper functions

def check_dates(df):
    
    indices = df.reset_index()['Indices']
    new_df = df.reset_index().drop(['Indices'], axis = 1).set_index(['Name']).T.reset_index().rename(columns = {'index': 'Date'})

    # new_df['Date'] = pd.to_datetime(new_df['Date'])

    sbin = get_history(symbol='SBIN', start = date.today() - timedelta(days = 7),end = date.today())
    latest_date_data = pd.to_datetime(new_df["Date"]).max().date()
    latest_date_available = sbin.index.max()

    if latest_date_available == latest_date_data:
        return True
    else:
        return [latest_date_data + timedelta(days = 1), latest_date_available]

def get_stocks(df, start_date, end_date):

    print("\n Downloading stocks. Please wait....")
    indices = df.reset_index()['Indices'].to_list()
    df = df.reset_index().drop(['Indices'], axis = 1).set_index(['Name'])
    for i, row in tqdm.tqdm(df.iterrows(), total = df.shape[0]):
        data = get_history(symbol = i, start = start_date, end = end_date)
        temp_dict = dict(zip(data.index.to_list(), data.Close.to_list()))
        for key, val in temp_dict.items():
            df.loc[i, key] = val
    df['Indices'] = indices
    return df.reset_index().set_index(['Name'])

def update_stocks(date_range, df):
    
    if date_range == True:
        print("Historical data already up to date")
        return True
    else:
        start_date = date_range[0]
        end_date = date_range[1]
        return get_stocks(df, start_date, end_date)

def get_all_stocks(file_name):
    
    """ 
    Input: Name of file which has Index and its correspoding csv file location which has list of stock names
    Output: A dictionary with key = stock_name and value = [indexes of stock] 
    """

    index_df = pd.read_csv(file_name)
    index_urls = dict(zip(index_df.Index, index_df.Link))

    stock_index_dict = defaultdict(list)

    for index, link in index_urls.items():
        s = requests.get(link).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        file_name = "index_stocks_" + str(index) + ".csv"
        for stock in df.Symbol.to_list():
            stock_index_dict[stock].append(index)
    
    return stock_index_dict


def clean_df_transpose(data_df, weeks_lookback):

    """
    Input: dataframe with stock (as rows) and timestamp (as columns) with cell values = closing stock price
    Output: clean database which is a transpose of input without index list. Stocks that do not have sufficient data are dropped
    """

    all_stocks = data_df.reset_index()['Name'].to_list()
    ### Dropping stocks that are NaN
    data_df = data_df.reset_index().set_index(['Name'])
    all_stocks = data_df.index.to_list()
    dropped_stocks = data_df[data_df.drop(['Indices'], axis = 1).isna().all(axis=1)].index.to_list()
    data_df = data_df[~data_df.drop(['Indices'], axis = 1).isna().all(axis=1)]

    ### Dropping stocks with not enough data (last 4 months)

    days = weeks_lookback * 7

    data_df = data_df.drop(['Indices'], axis = 1).T.reset_index().rename(columns = {"index" : "Date"})
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    data_df.sort_values(by=['Date'], inplace = True, ascending = False)
    data_df = data_df.head(days)

    dropped_stocks.extend(data_df.columns[data_df.isna().any()].tolist())

    final_stock_list = ['Date'] + list(set(all_stocks) - set(dropped_stocks)) 

    data_df = data_df[final_stock_list].set_index(['Date'])

    print("\n\nDropping stocks with not enough data:")
    for stock in dropped_stocks:
        print(stock)

    return data_df

def plot_stock_trends(stock, df, avg_return, dev_return, results):

    """
    Input:  stock = stock_name
            df = n x 1 dataframe with index as date and column as close price of 'stock'
            avg_return = average return performance of stock
            dev_return = deviation of list of returns of stock for a set period of time
            results = [timestamps of return calculation, % return corresponding to timestamp]

    Output: Shows stock performance in graphs
    """

    res = 'Avg Return: %s%% | Dev Return: %s%%'%(round(100*avg_return,2), round(100*dev_return,2))

    df.index = [d.date() for d in df.index]
    fig = plt.figure(figsize=(15,10))

    plt.subplot(2,1,1)
    plt.plot(df)
    plt.title(stock, fontsize=16)
    plt.ylabel('Price (₹)', fontsize=14)

    plt.subplot(2,1,2)
    plt.plot(results[0], results[1], color='g')
    plt.title(res, fontsize=16)
    plt.ylabel('%age Return', fontsize=14)
    plt.axhline(0, color='k', linestyle='--')

    plt.tight_layout()
    return fig

def analyse_stock(df, return_period, verbose = False):

    """
    Input:  df = n x 1 dataframe with index as date and column as close price of a stock
            return_period = size of sliding window to calculate return
    
    Output: avg_return = average return performance of stock
            dev_return = deviation of list of returns of stock for a set period of time
            results = [timestamps of return calculation, % return corresponding to timestamp]
    """

    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()

    df = df.reset_index()
    
    pct_return_after_period = []
    buy_dates = []

    for i, row in df.iterrows():

        buy_date = row['Date']
        buy_price = df[df.index == i].iloc[:,1].iloc[0]
        sell_date = buy_date + timedelta(weeks = return_period)
        
        try:
            sell_price = df[df.Date == sell_date].iloc[:,1].iloc[0]
        
        except IndexError:
            continue

        pct_return = (sell_price - buy_price)/buy_price
        pct_return_after_period.append(pct_return)
        buy_dates.append(buy_date)

        if verbose:

            print('Date Buy: %s, Price Buy: %s'%(buy_date,round(buy_price,2)))
            print('Date Sell: %s, Price Sell: %s'%(sell_date,round(sell_price,2)))
            print('Return: %s%%'%round(pct_return*100,1))
            print('-------------------')

    return np.mean(pct_return_after_period), np.std(pct_return_after_period), [buy_dates, pct_return_after_period]