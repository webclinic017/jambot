import operator as opr
from pickle import NONE
import sys
from datetime import datetime as dt
from datetime import timedelta as delta

import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from matplotlib import pyplot
from plotly.subplots import make_subplots

from . import functions as f

colors = dict(
    lightblue='#6df7ff',
    lightred='#ff6d6d'
)


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
    
    fig.add_trace(go.Scatter(x=df.Timestamp, y=df[f'ema{fast}'], line=dict(color='#18f27d', width=1)))
    fig.add_trace(go.Scatter(x=df.Timestamp, y=df[f'ema{slow}'], line=dict(color='#9d19fc', width=1)))
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
            range=[0,0.4]))
    fig.update_xaxes(
        autorange=True,
        tickformat=f.time_format(hrs=True),
        gridcolor='#e6e6e6',
        showgrid=False,
        gridwidth=1,
        tickangle=315)
    
    fig.show()

def add_traces(df, fig, traces):
    """Create traces from list of dicts and add to fig"""
    def _append_trace(trace, m):
        fig.append_trace(
            trace=trace,
            row=m.get('row', 1),
            col=m.get('col', 1))

    for m in traces:
        m['df'] = m.get('df', df) # allow using different df (for balance)
        traces = m['func'](**m)
        if isinstance(traces, list):
            for trace in traces:
                _append_trace(trace, m)
        else:
            _append_trace(traces, m)

def candlestick(df, **kw):
    return go.Candlestick(
        name='candles',
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close,
        increasing=dict(line=dict(color=colors['lightblue'])),
        decreasing=dict(line=dict(color=colors['lightred'])),
        line=dict(width=1),
        hoverlabel=dict(split=False, bgcolor='rgba(0,0,0,0)'))

def scatter(df, name, color, hoverinfo='all', **kw):
    return go.Scatter(
        name=name,
        x=df.index,
        y=df[name],
        mode='lines',
        line=dict(color=color, width=1),
        hoverinfo=hoverinfo)

def predictions(df, name, **kw):
    """Add traces of predicted 1, 0, -1 vals as shapes"""
    df = df.copy()
    # TODO 200 needs to be a %, not hard coded
    df['sma_low'] = df.Low.rolling(10).mean() - 200
    df['sma_low'] = df[['sma_low', 'Low']].min(axis=1) - 200
    df['sma_low'] = df['sma_low'].ewm(span=5).mean()
    df['sma_low'] = df[['sma_low', 'Low']].min(axis=1) - 200

    m = {
        'triangle-up': [1, 'green'],
        'circle': [0, '#8c8c8c'],
        'triangle-down': [-1, 'red']}
        
    traces = []

    for name, v in m.items():
        df2 = df[df.y_pred == v[0]]

        # filter to prediction correct/incorrect
        m3 = {'': opr.eq, '-open': opr.ne}
        for shape_suff, op in m3.items():
            df3 = df2[op(df2.y_pred, df2.target)]

            # set hover label to proba % text
            fmt = lambda x, y: f'{y}: {x:.1%}'
            text = df3.index.strftime('%Y-%m-%d %H') + '<br>' + \
                df3.proba_long.apply(fmt, y='long') + '<br>' + \
                df3.proba_short.apply(fmt, y='short')

            trace = go.Scatter(
                name='preds',
                x=df3.index,
                y=df3.sma_low,
                mode='markers',
                marker=dict(
                    size=4,
                    color=v[1],
                    symbol=f'{name}{shape_suff}'),
                text=text,
                hoverinfo='text')

            traces.append(trace)
    
    return traces

def probas(df, **kw):
    df = df.copy()
    df.proba_short = df.proba_short * -1
    traces = []

    names = dict(
        long=colors['lightblue'],
        short=colors['lightred']
        )

    for name, color in names.items():
        trace = go.Bar(
            name=name,
            x=df.index,
            y=df[f'proba_{name}'],
            marker=dict(
                line=dict(width=0),
                color=color),
            offsetgroup=0
        )
        
        traces.append(trace)

    return traces

def chart(df, symbol='XBTUSD', periods=200, last=True, startdate=None, df_balance=None):

    # py.init_notebook_mode(connected=False)

    traces = [
        dict(name='ema10', func=scatter, color='orange', hoverinfo='skip'),
        dict(name='ema50', func=scatter, color='#18f27d', hoverinfo='skip'),
        dict(name='ema200', func=scatter, color='#9d19fc', hoverinfo='skip'),
        dict(name='y_pred', func=predictions),
        dict(name='candle', func=candlestick),
        dict(name='probas', func=probas, row=2),
        # dict(name='vty_ema', func=scatter, color='green', row=2),
        # dict(name='vty_sma', func=scatter, color='yellow', row=2),
        # dict(name='rsi', func=scatter, color='purple', row=2),
        # dict(name='rsi_stoch', func=scatter, color='red', row=2),
        # dict(name='rsi_stoch_k', func=scatter, color='blue', row=2),
        # dict(name='rsi_stoch_d', func=scatter, color='green', row=2),
        ]
    
    if not df_balance is None:
        traces.append(dict(name='balance', func=scatter, color='#91ffff', row=3))
        df = df.merge(right=df_balance, how='left', left_index=True, right_index=True) \
            .assign(balance=lambda x: x.balance.fillna(method='ffill'))
        # display(df.tail(20))

    if startdate:
        df = df[df.index >= startdate] \
            .iloc[:periods, :]
    elif last:
        df = df.iloc[-1 * periods:, :]
    else:
        df = df.iloc[:periods, :]
    
    rows = max([m.get('row', 1) for m in traces])
    row_widths = [0.2, 0.2, 0.6][rows * -1:]
    height = 1000 * sum(row_widths)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_width=row_widths)
    fig = go.FigureWidget(fig)

    def _zoom(layout, xrange, *args, **kw):
        print('changing')
        in_view = df.loc[fig.layout.xaxis.range[0]: fig.layout.xaxis.range[1]]
        fig.layout.yaxis.range = [in_view.Low.min() - 10, in_view.High.max() + 10]
    
    fig.layout.on_change(_zoom, 'xaxis.range')

    add_traces(df, fig, traces)

    xaxis = dict(
        tickformat=f.time_format(hrs=True),
        rangeslider_thickness=0.05,
        tickangle=315)

    # update all axes in subplots
    fig.update_xaxes(
        gridcolor='#182633',
        autorange=True,
        fixedrange=False)
    fig.update_yaxes(
        gridcolor='#182633',
        side='right',
        autorange=True,
        fixedrange=False)

    fig.update_layout(
        xaxis=xaxis,
        height=height,
        # hovermode='x unified',
        # width=1000,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title=symbol)

    return fig
 
    # if not df2 is None:
    #     df3 = df2.loc[df2['PercentChange'] < 0]
    #     percentTrace = go.Bar(x=df3['Timestamp'], 
    #             y=df3['PercentChange'], 
    #             marker=dict(line=dict(width=2, color='#ff7a7a')))
    #     fig.append_trace(percentTrace, row=2, col=1)
        
    #     df3 = df2.loc[df2['PercentChange'] >= 0]
    #     percentTrace = go.Bar(x=df3['Timestamp'], 
    #             y=df3['PercentChange'], 
    #             marker=dict(line=dict(width=2, color='#91ffff')))
    #     fig.append_trace(percentTrace, row=2, col=1)

    #     balanceTrace = go.Scatter(
    #                     x=df2['Timestamp'],
    #                     y=df2['Balance'],
    #                     mode='lines',
    #                     line=dict(color='#91ffff', width=1))
    #     fig.append_trace(balanceTrace, row=3, col=1)
