import plotly.express as px
import pandas as pd

def plot_anomaly_3d(df: pd.DataFrame):
    """
    Generates an interactive 3D scatter plot of flagged anomalies.
    """
    fig = px.scatter_3d(
        df,
        x='amount',
        y='oldbalanceOrg',
        z='newbalanceOrg',
        color='anomaly_score',
        color_continuous_scale='Reds_r',
        title='#D Transaction Topogrpahy (Critical Anomalies)',
        opacity=0.8,
        labels={
            'amount': 'Transaction Amount',
            'oldbalanceOrg': 'Origin Old Balance',
            'newbalanceOrg': 'Origin New Balance',
            'anomaly_score': 'Severity'
        }
    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    return fig