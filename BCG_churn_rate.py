from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
#from mlxtend.frequent_patterns import fpgrowth
#import polars as pl
import os
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def import_data()-> pd.DataFrame:
    df = pd.read_csv(r"C:\Users\cesar\Dropbox\My PC (LAPTOP-GU3S2J8B)\Downloads\BCG\transactions_dataset.csv",sep =';',dtype={
    'client_id': 'int',
    'product_id': 'int',
    'quantity': 'int',
    'branch_id': 'int',
})
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop('product_id', axis =1)
    return df


def order_history(df: pd.DataFrame) -> pd.DataFrame:
    df['date_order'] = pd.to_datetime(df['date_order'])
    df['week'] = df['date_order'].dt.isocalendar().week
    df = df.sort_values(['client_id', 'date_order'])
    
    # create a new dataframe to hold the time deltas
    df['time_diff'] = df.groupby('client_id')['date_order'].diff().fillna(pd.Timedelta(0))
    # add a new column to the dataframe with the number of weeks since each client's last transaction
    df['weeks_since_last_transaction'] = df['time_diff'].dt.days /7

    # fill any missing values with 0
    df['weeks_since_last_transaction'] = df['weeks_since_last_transaction'].fillna(0)
    df['avg_time_between_transactions'] = df.groupby('client_id')['weeks_since_last_transaction'].transform('mean')
    df['max_time_between_transactions'] = df.groupby('client_id')['weeks_since_last_transaction'].transform('max')
    return df

def churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes churn rate for each client based on their transaction history.
    A client churned if he last bought something more than 3x the average transaction time or 1.5x the max transaction time 

    Parameters:
    -----------
    df: The input dataframe containing transaction data for each client.

    Returns:
    --------
    df: The dataframe with additional columns for churn rate calculation.
    """
    df = order_history(df)
    df['churn'] = np.where((df['weeks_since_last_transaction'] > 3 * df['avg_time_between_transactions']) |
                           (df['weeks_since_last_transaction'] > 1.5 * df['max_time_between_transactions']), 1, 0)
   
    return df
def churn_rate(df):
    rate = df['churn'].sum()/df['churn'].count()
    return rate

def order_invoice_diff(df):
    """
    Calculates the number of days between the invoice and the order date 
    Parameters:
    -----------
    df: The input dataframe containing the 'date_invoice' and 'date_order' columns.

    Returns:
    --------
    df: The dataframe with the additional 'order_invoice_diff' column.
    """

    df['order_invoice_diff'] = (pd.to_datetime(df['date_invoice']) - pd.to_datetime(df['date_order'])).dt.days
    return df

def weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = churn(df)
    df.drop(['date_invoice','order_channel','week'], axis =1)
    df = df.groupby(["client_id", pd.Grouper(key="date_order",freq='W')]).agg({
    "quantity": "sum",
    "sales_net": "sum",
    'churn':'max',
    'weeks_since_last_transaction':'max',
    'avg_time_between_transactions':'mean'  ,
    'max_time_between_transactions': 'max', 
    'order_invoice_diff':'max'  
}).reset_index()
    df["total_quantity_per_id"] = df.groupby(["client_id"])["quantity"].transform("sum")
    df["total_sales_id"] = df.groupby(["client_id"])["sales_net"].transform("sum")
    return df

def get_last_churn_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the last date for each client_id where churn is 0 and the last date where churn is 1.
    If churn is never 1 for a client_id, it returns the last date for that client_id.

    Parameters:
    -----------
    df: The input dataframe containing transaction data for each client.

    Returns:
    --------
    result:  The dataframe with the last date for each client_id where churn is 0 and the last date where churn is 1.
        If churn is never 1 for a client_id, it returns the last date for that client_id.
    """
    df = weekly_data(df)
    # Get the last date where churn is 0 for each client_id
    last_churn_0 = df.loc[df["churn"] == 0].groupby("client_id").apply(lambda x: x.iloc[-1])

    # Get the last date where churn is 1 for each client_id
    last_churn_1 = df.loc[df["churn"] == 1].groupby("client_id").apply(lambda x: x.iloc[-1])

    # Concatenate the two dataframes and drop duplicates
    result = pd.concat([last_churn_0, last_churn_1]).drop_duplicates()

    # Get the last date for each client_id where churn is never 1
    missing_client_ids = set(df["client_id"].unique()) - set(result["client_id"].unique())
    if missing_client_ids:
        missing_rows = df.groupby("client_id").apply(lambda x: x.iloc[-1]).loc[missing_client_ids]
        result = pd.concat([result, missing_rows])

    return result

df = import_data()
df = weekly_data(df)
df_last = get_last_churn_dates(df)
