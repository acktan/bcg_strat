import plotly.graph_objects as go
import pandas as pd

def plot_ts(ts, date_dimension="year/week", to_plot_dimension="sales_net"):
    to_plot = ts.copy()

    if date_dimension=="year/week":
        to_plot["date_order"] = pd.to_datetime(to_plot.date_order)
        to_plot[date_dimension] = to_plot.date_order.apply(lambda x: f'{x.year}/{x.week}')
        to_plot2 = to_plot.copy()
        to_plot2 = to_plot2[["year/week", "quantity", "order_channel", "sales_net"]]\
            .groupby(["order_channel", "year/week"])\
            .sum()
        to_plot2 = to_plot2.reset_index()
    else:
        print("Unaccepted 'date_dimension'.")
        
    fig = go.Figure()
    for channel in to_plot2.order_channel.unique():
        x = to_plot2[to_plot2.order_channel==channel]["year/week"]
        y = to_plot2[to_plot2.order_channel==channel][to_plot_dimension]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=channel))

    fig.update_layout(yaxis_title=to_plot_dimension, width=800, height=400)
    fig.update_layout(xaxis_title='Date (aggregated per week)', width=800, height=400)

    return fig