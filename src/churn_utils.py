from dateutil.relativedelta import relativedelta
import pandas as pd
import datetime


def preprocess_client_transactions(df_transaction, df_relations, client_id, churn_date):
    '''
    Preprocess the transaction data to feed the machine learning model. 

    Args:
        df : pd.DataFrame = transaction dataframe
        client_id : int = client identifier
        churn_date : str = date of the churn data at format '%Y-%m-%d'
    Output:
        all_features : pd.DataFrame = row of the features corresponding to this client and this churn information.
    '''
    # Convert dates
    churn_date = datetime.datetime.strptime(churn_date, '%Y-%m-%d')
    
    # Select data
    sample_df = df_transaction[df_transaction.client_id==client_id].copy()

    end_date = churn_date
    start_date = end_date - relativedelta(months=6)
    sample_df = sample_df[(sample_df.date_order>start_date) * (sample_df.date_order<churn_date)]

    # Extract purchases only (no reimbursement)
    sample_df_purchases = sample_df[sample_df.sales_net>0]

    # Add statistics on sales
    delta_t_list = [2, 24] # in week 
    all_features = pd.DataFrame(data={'client_id': [client_id]})

    for delta_t in delta_t_list:
        start_t = churn_date - relativedelta(weeks=delta_t)
        df = sample_df_purchases[sample_df_purchases.date_order>start_t][["sales_net", "client_id", "date_order"]].copy()

        # Add average and standard deviation of sales_net
        features_delta_t = df[["sales_net", "client_id"]]\
            .groupby("client_id")\
            .agg(['mean', 'std'])\
            .rename(columns={'mean': f'mean_{delta_t}', 'std': f'std_{delta_t}'})['sales_net']\
            .reset_index()\

        all_features[[f'mean_{delta_t}', f'std_{delta_t}']] = features_delta_t[[f'mean_{delta_t}', f'std_{delta_t}']]

        # Add mean time between two orders
        df_time_diff = df[['date_order']].sort_values('date_order', ascending=False).groupby('date_order').sum().reset_index().copy()
        df_time_diff["delta_time"] = df_time_diff.date_order - df_time_diff.date_order.shift(1)

        all_features[f'mean_delta_time_{delta_t}'] = df_time_diff.delta_time.mean()
        
        # Add the standard deviation of the time between two orders
        if delta_t==24:
            all_features[f'std_delta_time'] = df_time_diff.delta_time.std()

    # Add channel ratios
    channel_list = ['at the store', 'by phone', 'online', 'other', 'during the visit of a sales rep']
    for channel in channel_list:
        all_features[f'{channel}'] = (sample_df_purchases.order_channel==channel).sum()/len(sample_df_purchases)

    # Add relationship with client
    all_features["relationship"] = df_relations[df_relations.client_id==client_id].iloc[0,:].at['quali_relation']

    # Add number of products bought
    all_features["n_products"] = sample_df_purchases.product_id.nunique()

    # Add mean time between order and invoice
    all_features["mean_diff_order_invoice"] = (sample_df_purchases.date_invoice - sample_df_purchases.date_order).mean()

    # Add ratio of reimbursements
    all_features["reimbursement_ratio"] = (len(sample_df)-len(sample_df_purchases))/len(sample_df_purchases)

    return all_features