import plotly.graph_objects as go
import pandas as pd

def plot_ts(ts, date_dimension="year/week", to_plot_dimension="sales_net"):
    # Preprocesse
    to_plot = ts.copy()
    to_plot["date_order"] = pd.to_datetime(to_plot.date_order)
    
    if date_dimension=="year/week":
        to_plot[date_dimension] = to_plot.date_order.apply(lambda x: f'{x.year}/{x.week}')
    elif date_dimension in ["year", "month"]:
        to_plot[date_dimension] = to_plot.date_order.apply(lambda x: x[f"{date_dimension}"])
    else:
        print("Unaccepted 'date_dimension'.")

    to_plot = to_plot[[f"{date_dimension}", f"{to_plot_dimension}", "order_channel"]]\
        .groupby(["order_channel", f"{date_dimension}"])\
        .sum()\
        .reset_index()

    # Create plot 
    fig = go.Figure()
    for channel in to_plot.order_channel.unique():
        x = to_plot[to_plot.order_channel==channel]["year/week"]
        y = to_plot[to_plot.order_channel==channel][to_plot_dimension]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=channel))

    fig.update_layout(yaxis_title=to_plot_dimension, width=800, height=400)
    fig.update_layout(xaxis_title='Date (aggregated per week)', width=800, height=400)

    return fig