### Importing libraries and functions

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

### Helper functions from functions.py

from functions import *


### Main program functions

def display_options():
    option_dict = {1:"Load historical data", 2:"Check and update historical data", 3:"Run stock analysis", 4:"Display results of selected stocks", 5:"Save results", 6:"Exit"}
    print("=======================================")
    for i, option in option_dict.items():
        print(i, option)

def new_dataset():
    user_input = input("\nPlease enter file name containing urls of indices (should be present in current dataset). Program will exit if invalid file name is provided:\t")
    try:
        stock_index_dict = get_all_stocks(user_input)
    except Exception as e:
        print("\nError occured: ", e)
        sys.exit(0)
    
    data_df = pd.DataFrame({"Name": list(stock_index_dict.keys()), "Indices": list(stock_index_dict.values())}).set_index(['Name'])
    end_date = date.today()
    start_date = end_date - relativedelta(years = 2)

    return get_stocks(data_df, start_date = start_date, end_date = end_date)

def start():
    print("\n\nWelcome to stock analysis")
    print("\nTrying to load an existing historical dataset")

    try:
        data_df = pd.read_pickle('historical_data_new.pkl')
        print("Historical dataset loaded.")
    
    except:
        user_option_new_dataset = 'z'
        while(user_option_new_dataset not in ['y', 'n']):
            user_option_new_dataset = input("\nHistorical data could not be loaded.\nFile may be missing.\nDo you want to download and create new dataset for last 2 years? y/n:\t")

        if user_option_new_dataset == "y":
            data_df = new_dataset()

        elif user_option_new_dataset == 'n':
            sys.exit(0)
    
    print("Checking and updating historical dataset")
    
    if check_dates(data_df) == True:
        print("Dataset already up to date")
        data_df.to_pickle('historical_data_new.pkl')
    else:
        data_df = update_stocks(check_dates(data_df), data_df)
        print("Saving new dataset")
        data_df.to_pickle('historical_data_new.pkl')
    
    print("Performing stock analysis on historical dataset..")

    lookback = int(input("Enter lookback period (in weeks) for analysis. Enter 12 as default:"))
    return_period = int(input("Enter return period (in weeks). Enter 4 as default:"))
    min_avg_return = float(input("Enter minimum average return by share. Enter 0.08 as default:"))
    max_dev_return = float(input("Enter maximum deviation by share. Enter 0.07 as default:"))

    clean_df = clean_df_transpose(data_df = data_df, weeks_lookback = lookback + return_period)
    stock_list = clean_df.columns.to_list()
    stock_results_dict = {stock:{} for stock in stock_list}

    print("\nProcessing dataset...")

    for stock in tqdm.tqdm(stock_list):
        avg_return, dev_return, all_returns = analyse_stock(clean_df[[stock]], return_period, verbose = False)
        stock_results_dict[stock] = {"avg_return": avg_return, "dev_return": dev_return, "results": all_returns}

    print("\nProcessing dataset complete")

    print("===================================================================")

    print("Showing results for those stocks that meet the criteria:\n")

    selected_stocks = []
    for stock, stock_dict in stock_results_dict.items():
        if stock_dict['avg_return'] > min_avg_return and stock_dict['dev_return'] < max_dev_return:
            selected_stocks.append(stock)

    selected_stocks_dict = {k: stock_results_dict[k] for k in selected_stocks}
    print("Total number of stocks meeting the criteria: ", len(selected_stocks))
    print("List of these stocks:")
    for stock, stock_dict in selected_stocks_dict.items():
        print("---------------------")
        print("Stock name: ", stock)
        print("Average return: %s%%"%(round(stock_dict['avg_return']*100, 2)))
        print("Deviation of return: %s%%"%(round(stock_dict['dev_return']*100, 2)))
    print("---------------------")
    plot_dict = {}
    for stock, stock_dict in selected_stocks_dict.items():
        plt = plot_stock_trends(stock, clean_df[[stock]], stock_dict['avg_return'], stock_dict['dev_return'], stock_dict['results'])
        plot_dict[stock] = plt
    
    print("\nSaving plots to 'stock_analysis.pdf'")
    pdf_save = PdfPages('stock_analysis.pdf')

    for stock, plt in plot_dict.items():
        pdf_save.savefig(plt)

    pdf_save.close()

    print("'stock_analysis.pdf' saved")



if __name__ == "__main__":
    start()