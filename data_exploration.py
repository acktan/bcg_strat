from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
import polars as pl
import os

os.chdir(r'C:\Users\ckunt\OneDrive\Documents\Masters work\HEC\20. BCG\bcg_strat')
data = pl.read_parquet("data/transaction_data.parquet")
data_part = data
data_part = data_part.lazy().with_columns(pl.col(['date_order', 'date_invoice']).str.strptime(pl.Date, "%Y-%m-%d")).collect()


def detect_churn(df):
    now = df['date_order'].max()

    three_months_ago = now - timedelta(days=90)

    latest_order_date = df.lazy().groupby('client_id').agg([pl.max('date_order')]).collect()

    q = latest_order_date.select(
            [
                pl.col("*"),
                pl.when(pl.col('date_order') > three_months_ago).then(False).otherwise(True)
        ]
    )
    return q

churned = detect_churn(data_part)

churned.lazy().groupby('date_order').agg(
    [
        pl.count('literal')
    ]
).sort('date_order')

churned_weekly = churned.sort('date_order').groupby_dynamic("date_order", every="7d").agg(pl.col("literal").count())
quantity_weekly = data_part.sort('date_order').groupby_dynamic('date_order', every='7d').agg(pl.col('quantity').mean())
sales_channel = data_part.lazy().groupby(['date_order', 'order_channel']).agg(pl.col('sales_net').sum()).collect()
quantity_channel = data_part.lazy().groupby(['date_order', 'order_channel']).agg(pl.col('quantity').sum()).collect()

plt.plot(churned_weekly['date_order'], churned_weekly['literal'])
plt.plot(quantity_weekly['date_order'], quantity_weekly['quantity'])


sales_channel_weekly = sales_channel.sort('date_order').groupby_dynamic('date_order', every='7d').agg(
    [
        pl.col('sales_net').filter(pl.col('order_channel') == 'online').mean().alias('online'),
        pl.col('sales_net').filter(pl.col('order_channel') == 'by phone').mean().alias('by phone'),
        pl.col('sales_net').filter(pl.col('order_channel') == 'at the store').mean().alias('at the store'),
        pl.col('sales_net').filter(pl.col('order_channel') == 'during the visit of a sales rep').mean().alias('during the visit of a sales rep')
    ]
    ).sort(('date_order'))

quantity_channel_weekly = quantity_channel.sort('date_order').groupby_dynamic('date_order', every='7d').agg(
    [
        pl.col('quantity').filter(pl.col('order_channel') == 'online').mean().alias('online'),
        pl.col('quantity').filter(pl.col('order_channel') == 'by phone').mean().alias('by phone'),
        pl.col('quantity').filter(pl.col('order_channel') == 'at the store').mean().alias('at the store'),
        pl.col('quantity').filter(pl.col('order_channel') == 'during the visit of a sales rep').mean().alias('during the visit of a sales rep')
    ]
    ).sort(('date_order'))


def plot_stackgraph(df: pl.DataFrame,
          title: str) -> type[plt.figure]:
    _, ax = plt.subplots()
    for cols in df.columns[1:]:
        ax.plot(df['date_order'], df[cols], label=cols)
    
    ax.legend()
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylabel('Quantity')
    plt.title(title)
    plt.show()

plot_stackgraph(quantity_channel_weekly,
                title='Average weekly quantity sold per sales channel')
plot_stackgraph(sales_channel_weekly,
                title='Average weekly net sales per channel')


### FP-Growth analysis ###
df  = data_part[0:10_000][["order_channel", "client_id", "quantity"]]
df = pd.DataFrame(df).T
df = df.rename(columns={0: "order_channel",
                        1: "client_id",
                        2: "quantity"})

basket = (df.groupby(["order_channel", "client_id"])["quantity"]
            .sum().unstack().reset_index().fillna(0)
            .set_index("order_channel"))

basket = basket.astype("bool")

frequent_item_sets = fpgrowth(basket, min_support=0.05, use_colnames=True)
frequent_item_sets.sort_values(by="support", ascending=False).head(10)

df.client_id.value_counts()


