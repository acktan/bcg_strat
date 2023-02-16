import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator

class LoadData():
    def __init__(self, 
                 transaction_path: str,
                 relationship_path: str) -> None:
        
        self.transaction_path = transaction_path
        self.relationship_path = relationship_path
        
    
    def get_data(self, 
                 percentile: int = 90) -> pd.DataFrame:
        """Load and preprocess data:
        
        Args:
            percentile: percentile of lapsed period to consider as cut-off for churned/not-churned
        Returns:
            df: transaction/client dataframe with churned values
        """
        self.df = pd.read_csv(self.transaction_path)
        relations = pd.read_csv(self.relationship_path)
        
        self.df['date_invoice'] = pd.to_datetime(self.df['date_invoice'])
        self.df['date_order'] = pd.to_datetime(self.df['date_order'])
        self.df = self.df.sort_values('date_order')
        self.df = self.df.merge(relations, on='client_id', how="left")
        self.df['lapsed_period'] = self.df.groupby('client_id')['date_order'].diff().dt.days.fillna(0)
        
        D = np.percentile(self.df['lapsed_period'], percentile)
        self.df['churned'] = np.where(self.df['lapsed_period'] > D, 1, 0)
        return self.df

    
    
class Model():
    def __init__(self) -> None:
        pass
    
    def preprocess_simple(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

        cat_cols = ['order_channel', 'quali_relation']

        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(df[cat_cols])

        cat_cols_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))

        df = pd.concat([df.drop(cat_cols, axis=1), cat_cols_encoded], axis=1)

        y = df['churned']
        X = df.drop(columns=['date_order', 'date_invoice', 'client_id', 'product_id', 'churned', 'lapsed_period'])
        
        return X, y
    
    def create_simple_model(self, df: pd.DataFrame) -> tuple[BaseEstimator, pd.DataFrame, pd.Series]:
        X, y = self.preprocess_simple(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

        model = ExtraTreesClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test

class Computing():
    def __init__(self,
                 df: pd.DataFrame) -> None:
        self.df = df.sort_values('date_order')
        
    def sales(self,
              client_id: int = None,
              product_id: int = None):
        
        if client_id == 0 and product_id == 0:
            df = self.df
            
        if client_id != 0:
            df = self.df[self.df['client_id'] == client_id]
        
        if product_id != 0:
            df = self.df[self.df['product_id'] == product_id]
        
        sales_weekly = df.groupby(pd.Grouper(key='date_order', freq='7d')).agg({'sales_net': 'mean'})
        sales_channel = df.groupby(['date_order', 'order_channel'])['sales_net'].sum(numeric_only=True).reset_index()
        sales_channel_weekly = sales_channel.groupby(['order_channel']).resample('W', on='date_order').sum()
        
        return sales_channel_weekly

        
    def quantity(self, 
                 client_id: int = None, 
                 product_id: int = None):
        
        if client_id == 0 and product_id == 0:
            df = self.df
            
        if client_id != 0:
            df = self.df[self.df['client_id'] == client_id]
        
        if product_id != 0:
            df = self.df[self.df['product_id'] == product_id]
            
        
        quantity_weekly = df.groupby(pd.Grouper(key='date_order', freq='7d')).agg({'quantity': 'mean'})
        quantity_channel = df.groupby(['date_order', 'order_channel'])['quantity'].sum(numeric_only=True).reset_index()
        quantity_channel_weekly = quantity_channel.groupby(['order_channel']).resample('W', on='date_order').sum()
        
        return quantity_channel_weekly
    
    @staticmethod
    def plot_stackgraph(df: pd.DataFrame,
                        title: str,
                        ylabel: str) -> type[plt.figure]:
        
        _, ax = plt.subplots()
        for cols in df.index.get_level_values('order_channel').unique():
            ax.plot(df.loc[cols, :], label=cols)
        
        ax.legend()
        plt.xlabel('Date')
        plt.xticks(rotation=90)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
        
    @staticmethod
    def plotly_stackgraph(df: pd.DataFrame, 
                          title: str,
                          ylabel: str,
                          to_plot_dimension: str) -> go.Figure:
        
        fig = go.Figure()
        df = df.reset_index()
        for channel in df.order_channel.unique():
            x = df[df.order_channel==channel]['date_order']
            y = df[df.order_channel==channel][to_plot_dimension]
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=channel))
                    
        fig.update_layout(yaxis_title=ylabel, width=800, height=400)
        fig.update_layout(xaxis_title='Date (aggregated per week)', width=800, height=400)
        fig.update_layout(title=title)
        
        return fig