from datetime import (datetime as dt, timedelta as delta)

import seaborn as sns
from matplotlib import pyplot
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots

from . import (
    functions as f)

# Pandas DataFrame Styles
def background_contracts(s):
    # color = 'red' if val < 0 else 'black'
    # return 'color: {}'.format(color)
    is_neg = s < 0
    return ['background-color: red' if v else 'background-color: blue' for v in is_neg]

def neg_red(s):
    return ['color: #ff8080' if v < 0 else '' for v in s]

# Pandas plot on 2 axis
# ax = sym.df.plot(kind='line', x='Timestamp', y=['ema50', 'ema200'])
# sym.df.plot(kind='line', x='Timestamp', y='conf', secondary_y=True, ax=ax)

# Charts
def matrix(df, cols):
    return df.pivot(cols[0], cols[1], cols[2])

def heatmap(df, cols, title='', dims=(15,15)):
    _, ax = pyplot.subplots(figsize=dims)
    ax.set_title(title)
    return sns.heatmap(ax=ax, data=matrix(df, cols), annot=True, annot_kws={"size":8}, fmt='.1f')

def plot_chart(df, symbol, df2=None):
    py.iplot(chart(df=df, symbol=symbol, df2=df2))

def add_order(order, ts, price=None):    
    if price is None:
        price = order.price
        if order.marketfilled:
            linedash = 'dash'
        else:
            linedash = None
    else:
        linedash = 'dot'
    
    ft = order.filledtime
    if ft is None:
        ft = ts + delta(hours=24)
        linedash = 'dashdot'
    elif ft == ts:
        ft = ts + delta(hours=1)
    
    if not order.cancelled:
        if order.ordtype == 2:
            color = 'red'
        else:
            color = 'lime' if order.side == 1 else 'orange'
    else:
        color = 'grey'

    return go.layout.Shape(
        type='line',
        x0=ts,
        y0=price,
        x1=ft,
        y1=price,
        line=dict(
            color=color,
            width=2,
            dash=linedash))

def chart_orders(t, pre=36, post=None, width=900, fast=50, slow=200):
    dur = t.duration()
    df = t.df()
    if post is None:
        post = dur if dur > 36 else 36

    ts = t.candles[0].Timestamp
    timelower = ts - delta(hours=pre)
    timeupper = ts + delta(hours=post)
    
    mask = (df['Timestamp'] >= timelower) & (df['Timestamp'] <= timeupper)
    df = df.loc[mask].reset_index(drop=True)
    
    shapes, x, y, text = [], [], [], []
    for order in t.all_orders():
        shapes.append(add_order(order, ts=ts))
        if order.marketfilled or order.cancelled:
            shapes.append(add_order(order, ts=ts, price=order.pxoriginal))
        
        x.append(ts + delta(hours=-1))
        y.append(order.price),
        text.append('({:,}) {}'.format(order.contracts, order.name[:2]))

    labels = go.Scatter(x=x, y=y, text=text, mode='text', textposition='middle left')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'ema{fast}'],line=dict(color='#18f27d', width=1)))
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'ema{slow}'],line=dict(color='#9d19fc', width=1)))
    fig.add_trace(candlestick(df))
    fig.add_trace(labels)
    
    scale = 0.8

    fig.update_layout(
        height=750 * scale,
        width=width * scale,
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title='Trade {}: {}, {:.2%}, {}'.format(t.tradenum, t.conf, t.pnlfinal, t.duration()),
        shapes=shapes,
        xaxis_rangeslider_visible=False,
        yaxis=dict(side='right',
			showgrid=False,
			autorange=True,
			fixedrange=False) ,
        yaxis2=dict(showgrid=False,
            overlaying='y',
            autorange=False,
            fixedrange=True,
            range=[0,0.4])
            )
    fig.update_xaxes(
        autorange=True,
        tickformat=f.time_format(hrs=True),
        gridcolor='#e6e6e6',
        showgrid=False,
        gridwidth=1,
        tickangle=315)
    
    fig.show()

def candlestick(df):
    trace = go.Candlestick(
        x=df['Timestamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='#6df7ff')),
        decreasing=dict(line=dict(color='#ff6d6d')),
        line=dict(width=1),
        hoverlabel=dict(split=False, bgcolor='rgba(0,0,0,0)'))
    return trace

def chart(df, symbol, df2=None):
    py.init_notebook_mode(connected=False)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[0.1, 0.1, 0.4])
    fig.append_trace(candlestick(df), row=1, col=1)
    
    # add ema cols 
    for col in df.columns:
        if 'ema' in col:
            if '50' in col: sColour = '#18f27d'
            if '200' in col: sColour = '#9d19fc'
            emaTrace = go.Scatter(
                x=df['Timestamp'],
                y=df[col],
                mode='lines',
                line=dict(color=sColour, width=1))
            fig.append_trace(emaTrace, row=1, col=1)

    df3 = df2.loc[df2['PercentChange'] < 0]
    percentTrace = go.Bar(x=df3['Timestamp'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#ff7a7a')))
    fig.append_trace(percentTrace, row=2, col=1)
    
    df3 = df2.loc[df2['PercentChange'] >= 0]
    percentTrace = go.Bar(x=df3['Timestamp'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#91ffff')))
    fig.append_trace(percentTrace, row=2, col=1)

    balanceTrace = go.Scatter(
                    x=df2['Timestamp'],
                    y=df2['Balance'],
                    mode='lines',
                    line=dict(color='#91ffff', width=1))
    fig.append_trace(balanceTrace, row=3, col=1)

    fig.update_layout(
        height=800,
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title=symbol + ' OHLC')
    fig.update_xaxes(
        autorange=True,
        tickformat=f.time_format(hrs=True),
        gridcolor='#e6e6e6',
        showgrid=True,
        gridwidth=1,
        tickangle=315)
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        side='right',
        row=1, col=1)
    fig.update_yaxes(
        showgrid=False,
        autorange=True,
        fixedrange=False,
        side='right',
        row=2, col=1)
    fig.update_yaxes(
        showgrid=False,
        autorange=True,
        fixedrange=False,
        side='right',
        row=3, col=1)

    return fig
 