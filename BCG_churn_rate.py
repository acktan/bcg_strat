from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
import polars as pl
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def import_data()-> pd.Dataframe:
    df = pd.read_csv(r"C:\Users\cesar\Dropbox\My PC (LAPTOP-GU3S2J8B)\Downloads\transactions_dataset.csv",sep =';',dtype={
    'client_id': 'int',
    'product_id': 'int',
    'quantity': 'int',
    'branch_id': 'int',
})
    return df

def clean_data(df: pd.Dataframe) -> pd.Dataframe:
    df.drop('product_id', axis =1)
    return df

def add_additional_data(df: pd.Dataframe) -> pd.Dataframe:
    df['total_quantity']=df.groupby('client_id')['quantity'].transform('sum')
    df['avg_quantity']=df.groupby('client_id')['quantity'].transform('mean')
    df['total_sales_net']=df.groupby('client_id')['sales_net'].transform('sum')
    df['quantity_by_week'] = df.groupby(['client_id', pd.Grouper(key='date_order', freq='W-MON')])['quantity'].transform('sum')

def order_history(df: pd.Dataframe) -> pd.Dataframe:
    df['date_order'] = pd.to_datetime(df['date_order'])
    df['week'] = df['date_order'].dt.isocalendar().week
    df = df.sort_values(['client_id', 'date_order'])
    
    # create a new dataframe to hold the time deltas
    time_diffs = df.groupby('client_id')['date_order'].diff().fillna(pd.Timedelta(0))
    # add a new column to the dataframe with the number of weeks since each client's last transaction
    df['weeks_since_last_transaction'] = df['time_diff'].dt.days /7

    # fill any missing values with 0
    df['weeks_since_last_transaction'] = df['weeks_since_last_transaction'].fillna(0)
    df['avg_time_between_transactions'] = df.groupby('client_id')['weeks_since_last_transaction'].transform('mean')
    df['max_time_between_transactions'] = df.groupby('client_id')['weeks_since_last_transaction'].transform('max')
    return df

def churn(df: pd.Dataframe) -> pd.Dataframe:
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
    df['churn']= np.where(df['weeks_since_last_transaction']>
                               3*df['avg_time_between_transactions'] or 1,5 *df['max_time_between_transactions'] ,1,0)
    return df