import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import sys
sys.path.append("../")
from helpers import LoadData, Model
from mlxtend.frequent_patterns import fpgrowth
from sklearn.metrics import confusion_matrix

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

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = pd.read_csv('../data/final_pred.csv')
if 'true' not in st.session_state:
    st.session_state['true'] = pd.read_csv('../data/true_values.csv')

predictions = st.session_state['prediction']
y_test = st.session_state['true']

loader = LoadData(transaction_path, relation_path)
if 'df' not in st.session_state:
    st.session_state['df'] = loader.get_data() ## test set
df = st.session_state['df']
df = df.merge(predictions, on='client_id', how='inner')


client_ids = np.insert(df['client_id'].unique(), 0, 0)
column_inputs = st.columns(2)
with column_inputs[0]:
    client_id = st.selectbox('Select a client ID:', options=client_ids, index=0)
with column_inputs[1]:
    threshold = st.slider(label='Adjust treshold sensitivity:',
                          min_value=0.0,
                          max_value=1.0,
                          step=0.05)
    

client_df = df[df['client_id'] == client_id]
churn = client_df['churn_prediction'].mean()

churn_columns = st.columns(2)
with churn_columns[0]:
    st.metric(label='Will churn?', value=np.where(churn > threshold, 1, 0))
    if churn > threshold:
        st.write('Customer {} is at risk of churning, here are some product bundles you can recommend to them'.format(client_id))
        # product recommendations
        client_df = df[df['client_id'] == client_id]
        basket = client_df.groupby(['product_id',pd.Grouper(freq='M', key='date_order')])['quantity']\
            .sum().unstack().reset_index().fillna(0).set_index('product_id').astype('bool')

        frequent_item_sets = fpgrowth(basket[0:50].T, min_support=0.01, use_colnames=True)
        frequent_item_sets = frequent_item_sets.sort_values(by="support", ascending=False)
        st.write(frequent_item_sets[frequent_item_sets['itemsets'].map(len) > 2].itemsets.head(3))
    else:
        st.write('This customer is not at risk of churning')
with churn_columns[1]:
    pred2 = []
    for i, row in predictions.iterrows():
        prob = row['churn_prediction']
        if prob > threshold:
            pred2 += [1]
        else:
            pred2 += [0]
    cf_matrix = confusion_matrix(y_test, pred2)
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap="Reds", fmt=".1%", cbar=False, ax=ax)

    # Set the axis labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Display the figure in Streamlit
    st.pyplot(fig)
    

    
#Output top 10 churners
st.subheader("Clients that are most at risk of churning:")
if 'topchurners' not in st.session_state:
    st.session_state['topchurners'] = df.groupby(['client_id'])['churn_prediction', 'sales_net']\
        .mean().sort_values(['churn_prediction', 'sales_net'], ascending=False).head(10)
        
topchurners = st.session_state['topchurners']
st.table(topchurners)




