import inspect
import operator as opr
import sys
from datetime import datetime as dt
from datetime import timedelta as delta

import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from matplotlib import pyplot
from plotly.subplots import make_subplots

from . import functions as f
from . import sklearn_helper_funcs as sf

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
    
    # set everything except candles to x2 so they dont show in rangeslider
    for trace in fig.data:
        if not trace['name'] == 'candles':
            trace.update(xaxis='x2')

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

def scatter(df, name, color=None, hoverinfo='all', stepped=False, **kw):

    # TODO make this more general than just for scatter
    signature = inspect.signature(go.Scatter.__init__)
    kw = {k: v for k, v in kw.items() if k in signature.parameters.keys()}

    kw['line'] = {**kw.get('line', {}), **dict(color=color, width=1)}
    if stepped:
        kw['line'] = {**kw.get('line'), **dict(shape='hv')}

    return go.Scatter(
        name=name,
        x=df.index,
        y=df[name],
        mode='lines',
        # line=dict(color=color, width=1),
        hoverinfo=hoverinfo,
        **kw)

def bar(df, name, color=None, **kw):
    trace = go.Bar(
        name=name,
        x=df.index,
        y=df[name],
        marker=dict(
            line=dict(width=0),
            color=color),
        offsetgroup=0,
    )
    return trace

def add_pred_trace(df, offset=2):
    # TODO this is so messy... 200 needs to be a %, not hard coded
    s_offset = df.Low * offset * 0.01

    # df['sma_low'] = df.Low.rolling(10).mean() - s_offset
    df['sma_low'] = (df.Low - s_offset).rolling(10).mean()
    df['sma_low'] = df[['sma_low', 'Low']].min(axis=1) #- s_offset
    # df['sma_low'] = df['sma_low'].ewm(span=5).mean()
    # df['sma_low'] = df[['sma_low', 'Low']].min(axis=1) - s_offset
    return df

def predictions(df, name, **kw):
    """Add traces of predicted 1, 0, -1 vals as shapes"""
    df = df.copy() \
        .pipe(add_pred_trace)

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

def trades(df, name, **kw):
    """Add traces of trade entries with direction and profitable|not"""
    df = df.copy() \
        .pipe(add_pred_trace, offset=4)

    m = {
        'star-triangle-up': [1, colors['lightblue']],
        'star-triangle-down': [-1, colors['lightred']]}
    
    traces = []

    for name, v in m.items():
        df2 = df[df.trade_side == v[0]]
        m2 = {'': opr.gt, '-open': opr.lt}

        for shape_suff, op in m2.items():
            df3 = df2[op(df2.trade_pnl, 0)]

            # set hover label
            fmt = lambda x, y: f'{y}: {x:,.0f}'
            text = df3.index.strftime('%Y-%m-%d %H') + '<br>' + \
                df3.trade_entry.apply(fmt, y='entry') + '<br>' + \
                df3.trade_exit.apply(fmt, y='exit')  + '<br>' + \
                df3.trade_pnl.apply(lambda x: f'pnl: {x:.2%}')

            trace = go.Scatter(
                name='trades',
                x=df3.index,
                y=df3.sma_low - 50,
                mode='markers',
                marker=dict(
                    size=8,
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

def chart(df, symbol='XBTUSD', periods=200, last=True, startdate=None, df_balance=None, traces=None):

    # py.init_notebook_mode(connected=False)
    if traces is None:
        traces = []

    traces = traces + [
        dict(name='ema10', func=scatter, color='orange', hoverinfo='skip'),
        dict(name='ema50', func=scatter, color='#18f27d', hoverinfo='skip'),
        dict(name='ema200', func=scatter, color='#9d19fc', hoverinfo='skip'),
        dict(name='y_pred', func=predictions),
        dict(name='candle', func=candlestick),
        # dict(name='probas', func=probas, row=2),
        # dict(name='vty_ema', func=scatter, color='green', row=2),
        # dict(name='vty_sma', func=scatter, color='yellow', row=2),
        # dict(name='rsi', func=scatter, color='purple', row=2),
        # dict(name='rsi_stoch', func=scatter, color='red', row=2),
        # dict(name='rsi_stoch_k', func=scatter, color='blue', row=2),
        # dict(name='rsi_stoch_d', func=scatter, color='green', row=2),
        ]
    
    # remove cols not in df
    include = ('candle', 'probas', 'trades')
    traces = [m for m in traces if m['name'] in df.columns or any(m['name'] == item for item in include)]

    if startdate:
        df = df[df.index >= startdate] \
            .iloc[:periods, :]
    elif last:
        df = df.iloc[-1 * periods:, :]
    else:
        df = df.iloc[:periods, :]
    
    rows = max([m.get('row', 1) for m in traces])
    row_widths = ([0.2] * (rows - 1) + [0.4])[rows * -1:]
    height = 1000 * sum(row_widths)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_width=row_widths,
        vertical_spacing=0.01)

    fig = go.FigureWidget(fig)

    add_traces(df, fig, traces)

    rangeselector = dict(
        font=dict(color='black'),
        buttons=[
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=14, label='14d', step='day', stepmode='backward'),
            dict(count=6, label='6d', step='day', stepmode='backward'),
            dict(count=2, label='2d', step='day', stepmode='backward'),
        ]
    )

    rng = [df.index[-14 * 24].to_pydatetime(), df.index[-2].to_pydatetime()]

    xaxis = dict(
        dtick='6h',
        tickformat=f.time_format(hrs=True),
        tickangle=315,
        rangeselector=rangeselector,
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
        side='top',
        showticklabels=True,
        range=rng
        )
    
    xaxis2 = dict(
        matches='x',
        overlaying='x',
        side='top',
        showgrid=False,
        showticklabels=False,
        range=rng
    )

    # update all axes in subplots
    fig.update_xaxes(
        range=rng,
        gridcolor='#182633',
        autorange=True,
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikethickness=1,
        showticklabels=True)
    fig.update_yaxes(
        gridcolor='#182633',
        side='right',
        autorange=True,
        fixedrange=False)

    fig.update_layout(
        xaxis=xaxis,
        xaxis2=xaxis2,
        xaxis_range=rng,
        xaxis2_range=rng,
        height=height,
        # width=1000,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        dragmode='pan',
        # title=symbol
        )
    
    # fig['layout']['xaxis'].update(range=rng)

    return fig
 
    # def _zoom(layout, xrange, *args, **kw):
    #     print('changing')
    #     in_view = df.loc[fig.layout.xaxis.range[0]: fig.layout.xaxis.range[1]]
    #     fig.layout.yaxis.range = [in_view.Low.min() - 10, in_view.High.max() + 10]
    
    # fig.layout.on_change(_zoom, 'xaxis.range')


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
