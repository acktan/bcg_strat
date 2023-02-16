import pandas as pd
import numpy as np
import streamlit as st
import sys
sys.path.append("../")
from helpers import LoadData, Model
from mlxtend.frequent_patterns import fpgrowth

import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title='Transaction exploration', page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Predicting customer churn')
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "Hardware Comp.";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 30px;
            position: relative;
            top: 100px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

transaction_path = '../data/transaction_data_short.csv'
relation_path = '../data/sales_client_relationship_dataset.csv'

loader = LoadData(transaction_path, relation_path)
df = loader.get_data() ## test set

client_ids = np.insert(df['client_id'].unique(), 0, 0)
client_id = st.selectbox('Select a client ID:', options=client_ids, index=0)
client_df = df[df['client_id'] == client_id]
churn = client_df['churned'].mean()

st.metric(label='Churn probability', value=f'{churn:20,.3f}')

if churn > 0.5:
    st.write('This customer is at risk of churning, here are some product bundles you can recommend to them')
else:
    st.write('This customer is not at risk of churning')

# product recommendations 
if churn > 0.5:
    client_df = df[df['client_id'] == client_id]

    basket = client_df.groupby(['product_id',pd.Grouper(freq='M', key='date_order')])['quantity']\
        .sum().unstack().reset_index().fillna(0).set_index('product_id').astype('bool')

    frequent_item_sets = fpgrowth(basket[0:50].T, min_support=0.01, use_colnames=True)
    frequent_item_sets = frequent_item_sets.sort_values(by="support", ascending=False)
    st.write('Customer {} has previously bought these items together in the past'.format(client_id))
    st.write(frequent_item_sets[frequent_item_sets['itemsets'].map(len) > 2].itemsets.head(3))
    
#Output top 10 churners
st.subheader("Clients that are most at risk of churning:")
topchurners = df.groupby(['client_id'])['churned', 'sales_net']\
    .mean().sort_values(['churned', 'sales_net'], ascending=False).head(10)
st.table(topchurners)




