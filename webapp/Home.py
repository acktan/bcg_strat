import numpy as np
import streamlit as st
import pandas as pd

from helpers import Model, LoadData, Computing

import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title='Transaction exploration', page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Closer look at client and branch transactions')

transaction_path = '../data/transaction_data_short.csv'
relation_path = '../data/sales_client_relationship_dataset.csv'

loader = LoadData(transaction_path, relation_path)
df = loader.get_data()

product_ids = np.insert(df['product_id'].unique(), 0, 0)
client_ids = np.insert(df['client_id'].unique(), 0, 0)

columns_input = st.columns(2)
with columns_input[0]:
    product_id = st.selectbox('Select a product ID:', options=product_ids, index=0)
with columns_input[1]:
    client_id = st.selectbox('Select a client ID:', options=client_ids, index=0)

compute = Computing(df)
sales_values = compute.sales(client_id, product_id)
quant_values = compute.quantity(client_id, product_id)

st.subheader("Sales Metrics by Order Channel")
unique_vals = len(sales_values.index.get_level_values('order_channel').unique())
column_stats = st.columns(unique_vals)

for i, order in enumerate(sales_values.index.get_level_values('order_channel').unique()):
    with column_stats[i]:
        st.metric(label=order.capitalize(),
                  value=f"{sales_values.loc[order, :].mean().values[0]:.2e} €")
st.pyplot(compute.plot_stackgraph(sales_values, 
                        title='Average weekly net sales per channel',
                        ylabel='€'))

st.subheader("Sales Quantities by Order Channel")
unique_vals = len(quant_values.index.get_level_values('order_channel').unique())
column_stats_1 = st.columns(unique_vals)
for i, order in enumerate(quant_values.index.get_level_values('order_channel').unique()):
    with column_stats_1[i]:
        st.metric(label=order.capitalize(),
                  value=f"{quant_values.loc[order, :].mean().values[0]:.2e} €")
st.pyplot(compute.plot_stackgraph(quant_values, 
                        title='Average weekly quantity sold per sales channel',
                        ylabel='Count'))



