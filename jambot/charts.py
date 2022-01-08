import inspect
import operator as opr
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from icecream import ic
from matplotlib import pyplot
from pandas import DataFrame
from plotly.subplots import make_subplots

from jambot import SYMBOL, getlog
from jambot.config import COLORS as colors
from jambot.utils.mlflow import MlflowManager
from jgutils import pandas_utils as pu

log = getlog(__name__)

FMT_HRS = '%Y-%m-%d %H'

ic.configureOutput(prefix='')


# Pandas DataFrame Styles
def background_contracts(s):
    # color = 'red' if val < 0 else 'black'
    # return 'color: {}'.format(color)
    is_neg = s < 0
    return ['background-color: red' if v else 'background-color: blue' for v in is_neg]


def neg_red(s):
    return ['color: #ff8080' if v < 0 else '' for v in s]

# Pandas plot on 2 axis
# ax = bm.df.plot(kind='line', x='timestamp', y=['ema50', 'ema200'])
# bm.df.plot(kind='line', x='timestamp', y='conf', secondary_y=True, ax=ax)

# Charts


def matrix(df, cols):
    return df.pivot(cols[0], cols[1], cols[2])


def heatmap(df, cols, title='', dims=(15, 15)):
    _, ax = pyplot.subplots(figsize=dims)
    ax.set_title(title)
    return sns.heatmap(ax=ax, data=matrix(df, cols), annot=True, annot_kws={'size': 8}, fmt='.1f')


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

    ts = t.candles[0].timestamp
    timelower = ts - delta(hours=pre)
    timeupper = ts + delta(hours=post)

    mask = (df['timestamp'] >= timelower) & (df['timestamp'] <= timeupper)
    df = df.loc[mask].reset_index(drop=True)

    shapes, x, y, text = [], [], [], []
    for order in t.all_orders():
        shapes.append(add_order(order, ts=ts))
        if order.marketfilled or order.cancelled:
            shapes.append(add_order(order, ts=ts, price=order.pxoriginal))

        x.append(ts + delta(hours=-1))
        y.append(order.price),
        text.append('({:,}) {}'.format(order.qty, order.name[:2]))

    labels = go.Scatter(x=x, y=y, text=text, mode='text', textposition='middle left')

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.timestamp, y=df[f'ema{fast}'], line=dict(color='#18f27d', width=1)))
    fig.add_trace(go.Scatter(x=df.timestamp, y=df[f'ema{slow}'], line=dict(color='#9d19fc', width=1)))
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
                        fixedrange=False),
        yaxis2=dict(showgrid=False,
                    overlaying='y',
                    autorange=False,
                    fixedrange=True,
                    range=[0, 0.4]))
    fig.update_xaxes(
        autorange=True,
        tickformat=FMT_HRS,
        gridcolor='#e6e6e6',
        showgrid=False,
        gridwidth=1,
        tickangle=315)

    fig.show()


def add_traces(df: pd.DataFrame, fig: go.FigureWidget, traces: List[dict]) -> None:
    """Create traces from list of dicts and add to fig
    - trace must be created with a function"""
    def _append_trace(trace, m):
        fig.append_trace(
            trace=trace,
            row=m.get('row', 1),
            col=m.get('col', 1))

    for m in traces:
        m['df'] = m.get('df', df)  # allow using different df (for balance)
        traces = m.get('func', scatter)(**m)

        if isinstance(traces, list):
            for trace in traces:
                _append_trace(trace, m)
        else:
            _append_trace(traces, m)

    # set everything except candles to x2 so they dont show in rangeslider
    for trace in fig.data:
        if not trace['name'] == 'candles':
            trace.update(xaxis='x2')


def candlestick(df, **kw) -> go.Candlestick:
    return go.Candlestick(
        name='candles',
        x=df.index,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        increasing=dict(line=dict(color=colors['lightblue'])),
        decreasing=dict(line=dict(color=colors['lightred'])),
        line=dict(width=1),
        hoverlabel=dict(split=False, bgcolor='rgba(0,0,0,0)'))


def scatter(df, name, color=None, hoverinfo='all', stepped=False, mode='lines', **kw):
    # mode = lines or markers
    # marker=dict(size=0.5),
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
        mode=mode,
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
    s_offset = df.low * offset * 0.01

    # df['sma_low'] = df.low.rolling(10).mean() - s_offset
    df['sma_low'] = (df.low - s_offset).rolling(10).mean()
    df['sma_low'] = df[['sma_low', 'low']].min(axis=1)  # - s_offset
    # df['sma_low'] = df['sma_low'].ewm(span=5).mean()
    # df['sma_low'] = df[['sma_low', 'low']].min(axis=1) - s_offset
    return df


def predictions(df, name, regression=False, **kw) -> List[go.Scatter]:
    """Add 6 traces of predicted 1, 0, -1 vals as shapes"""
    df = df.copy() \
        .pipe(add_pred_trace)

    m = {
        'triangle-up': [1, 'green'],
        'circle': [0, '#8c8c8c'],
        'triangle-down': [-1, 'red']}

    traces = []

    for name, (side, color) in m.items():

        # convert pred pcts to 1/-1 for counting as 'correct' or not
        if regression:
            df.y_pred = np.where(df.y_pred > 0, 1, -1)
            df.target = np.where(df.target > 0, 1, -1)

        df2 = df[df.y_pred == side]

        # filter to prediction correct/incorrect
        m3 = {'': opr.eq, '-open': opr.ne}
        for shape_suff, op in m3.items():

            df3 = df2[op(df2.y_pred, df2.target)]

            # set hover label to proba % text
            fmt = lambda x, y: f'{y}: {x:.1%}'
            text = df3.index.strftime('%Y-%m-%d %H') + '<br>'

            if all(col in df.columns for col in ('proba_long', 'proba_short')):
                text = text + \
                    df3.proba_long.apply(fmt, y='long') + '<br>' + \
                    df3.proba_short.apply(fmt, y='short')

            trace = go.Scatter(
                name='preds',
                x=df3.index,
                y=df3.sma_low,
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
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

    for name, (side, color) in m.items():
        df2 = df[df.trade_side == side]
        oprs = [('', opr.gt), ('-open', opr.lt), ('-open', opr.eq)]

        for shape_suff, op in oprs:
            df3 = df2[op(df2.trade_pnl, 0)]

            # set hover label
            fmt = lambda x, y: f'{y}: {x:,.0f}'

            text = df3.index.strftime('%Y-%m-%d %H') + '<br>' + \
                df3.trade_entry.apply(fmt, y='entry') + '<br>' + \
                df3.trade_exit.apply(fmt, y='exit') + '<br>' + \
                df3.trade_pnl.apply(lambda x: f'pnl: {x:.2%}') + '<br>' + \
                df3.dur.apply(lambda x: f'dur: {x:.0f}')

            if op is opr.eq:
                color = colors['lightyellow']

            trace = go.Scatter(
                name='trades',
                x=df3.index,
                y=df3.sma_low - 50,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol=f'{name}{shape_suff}'),
                text=text,
                hoverinfo='text')

            traces.append(trace)

    return traces


def trace_extrema(df, name, **kw):
    """Add trace of indicators for high/low peaks"""
    m = dict(
        maxima=dict(symbol='triangle-down-open', color=colors['lightred']),
        minima=dict(symbol='triangle-up-open', color=colors['lightblue'])) \
        .get(name)

    trace = go.Scatter(
        name=name,
        x=df.index,
        y=df[name],
        mode='markers',
        marker=dict(size=5, symbol=m['symbol'], color=m['color'])
    )
    return [trace]


def split_trace(df, name, split_val=0.5, **kw):
    """Split trace into two at value, eg 0.5 for long/short traces"""
    df = df.copy()
    traces = []
    oprs = [opr.lt, opr.gt]
    clrs = [colors['lightblue'], colors['lightred']]

    for color, op in zip(clrs, oprs):
        df2 = df.copy()
        df2.loc[op(df[name], split_val)] = split_val  # set lower/higher values to 0.5

        trace = scatter(
            df=df2,
            name=name,
            color=color,
            # fill='tonexty'
        )
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


def clean_traces(cols, traces):
    """Remove cols not in df"""
    traces = traces or []
    include = ('candle', 'probas', 'trades')
    return [m for m in traces if m['name'] in cols or any(m['name'] == item for item in include)]


def enum_traces(traces, base_num=2):
    """Give traces a row number?"""
    return [{**m, **dict(row=m.get('row', None) or i + base_num)} for i, m in enumerate(traces)]


def chart(
        df: DataFrame,
        symbol: str = SYMBOL,
        periods: int = 200,
        last: bool = True,
        startdate: dt = None,
        df_balance: DataFrame = None,
        traces: List[dict] = None,
        default_range: Tuple[dt, dt] = None,
        secondary_row_width: float = 0.12,
        **kw) -> go.FigureWidget:
    """Main plotting func for showing main candlesticks with supporting subplots of features

    Parameters
    ----------
    df : DataFrame
        df with reqd cols
    symbol : str, optional
        default SYMBOL
    periods : int, optional
        plot only n periods, by default 200
    last : bool, optional
        plot LAST n periods, by default True
    startdate : dt, optional
        plot from start date, by default None
    df_balance : DataFrame, optional
        add trace of balance over time, by default None
    traces : List[dict], optional
        extra traces to show in addition to base traces, by default None
    default_range : Tuple[dt, dt], optional
        used to set default slider range, by default None
    secondary_row_width : float, optional
        make smaller/larger rows under main plot, by default 0.12

    Returns
    -------
    go.FigureWidget
        figure to show()
    """
    bgcolor = '#000B15'
    gridcolor = '#182633'

    base_traces = [
        dict(name='ema_10', func=scatter, color='orange', hoverinfo='skip'),
        dict(name='ema_50', func=scatter, color='#18f27d', hoverinfo='skip'),
        dict(name='ema_200', func=scatter, color='#9d19fc', hoverinfo='skip'),
        dict(name='y_pred', func=predictions, **kw),
        dict(name='candle', func=candlestick),
    ]

    # clean and combine base with extra traces
    base_traces = clean_traces(cols=df.columns, traces=base_traces)
    traces = clean_traces(cols=df.columns, traces=traces)
    traces = enum_traces(traces)
    traces = base_traces + traces

    if startdate:
        df = df[df.index >= startdate].iloc[:periods, :]
    elif last:
        df = df.iloc[-periods:, :]
    else:
        df = df.iloc[:periods, :]

    rows = max([m.get('row', 1) for m in traces])
    row_widths = ([secondary_row_width] * (rows - 1) + [0.4])[rows * -1:]
    height = 1000 * sum(row_widths)

    # Set subplot titles
    subplot_titles = {}
    for m in traces:
        subplot_titles[m.get('row', 1)] = m.get('name', '')

    subplot_titles[1] = ''  # main chart doesn't need title
    subplot_titles = [subplot_titles[k] for k in sorted(subplot_titles)]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_width=row_widths,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        # specs=specs
    )

    fig = go.FigureWidget(fig)

    add_traces(df, fig, traces)

    rangeselector = dict(
        font=dict(color='black'),
        activecolor='red',
        buttons=[
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=14, label='14d', step='day', stepmode='backward'),
            dict(count=6, label='6d', step='day', stepmode='backward'),
            dict(count=2, label='2d', step='day', stepmode='backward'),
        ],
    )

    rng = None if default_range is None else [
        df.index[-1 * default_range * 24].to_pydatetime(),
        df.index[-1].to_pydatetime()]

    xaxis = dict(
        type='date',
        dtick='6h',
        tickformat=FMT_HRS,
        tickangle=315,
        rangeslider=dict(
            bgcolor=bgcolor,
            visible=True,
            thickness=0.0125,
            # range=rng
        ),
        rangeselector=rangeselector,
        side='top',
        showticklabels=True,
        range=rng,
    )

    xaxis2 = dict(
        matches='x',
        overlaying='x',
        side='top',
        showgrid=False,
        showticklabels=False,
    )

    # update all axes in subplots
    fig.update_xaxes(
        gridcolor=gridcolor,
        # autorange=True,
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikethickness=1,
        showticklabels=True)
    fig.update_yaxes(
        zerolinewidth=0.25,
        zerolinecolor=gridcolor,
        gridcolor=gridcolor,
        side='right',
        autorange=True,
        fixedrange=False)

    fig.update_layout(
        xaxis=xaxis,
        xaxis2=xaxis2,
        height=height,
        # width=1000,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#021629',
        plot_bgcolor=bgcolor,
        font_color='white',
        showlegend=False,
        dragmode='pan',
        # title=symbol
    )

    # update subplot title size/position
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=9)
        i['x'] = 0.05

    # fig.update_xaxes(type='date', range=[df.index[0], df.index[30]])
    return fig


def plot_strat_results(
        df: DataFrame,
        df_balance: DataFrame = None,
        df_trades: DataFrame = None,
        startdate: dt = dt(2021, 1, 1),
        periods: int = 60 * 24,
        regression: bool = False) -> None:
    """Plot strategy results from single strat run

    Parameters
    ----------
    df : DataFrame
        main df with ~10 cols
    df_balance : DataFrame, optional
        balance from strat.wallet, by default None
    df_trades : DataFrame, optional
        trates from strat.df_trades(), by default None
    startdate : dt, optional
        filter df to start date, by default dt(2021, 1, 1)
    periods : int, optional
        filter df to n periods, by default 60*24
    regression : bool, optional
        by default False
    """
    df = df.astype(float)
    split_val = 0.5 if not regression else 0
    df = df.iloc[-periods:] if startdate is None else df.loc[startdate:]
    rolling_col = 'proba_long' if not regression else 'y_pred'

    traces = [
        dict(name=rolling_col, func=split_trace, split_val=split_val),
        dict(name='rolling_proba', func=split_trace, split_val=split_val),
    ]

    # merge balances to full df with forward fill
    if not df_balance is None:
        traces.append(dict(name='balance', func=scatter, color='#91ffff', stepped=True))

        df = df \
            .pipe(pu.left_merge, df_balance) \
            .assign(balance=lambda x: x.balance.fillna(method='ffill'))

    # merge trades to show entry/exits as triangles in main chart
    if not df_trades is None:
        traces.append(dict(name='trades', func=trades, row=1))

        rename_cols = dict(
            side='trade_side',
            pnl='trade_pnl',
            entry='trade_entry',
            exit='trade_exit')

        df = df.pipe(pu.left_merge, df_trades.rename(columns=rename_cols).set_index('ts'))

    # create candlestick chart and show
    # candle chart is VERY slow/crashes if np.float16 dtype
    fig = chart(
        df=df,
        periods=periods,
        last=True,
        startdate=startdate,
        df_balance=df_balance,
        traces=traces,
        regression=regression)

    fig.show()


def plot_runs_pnl():
    mfm = MlflowManager()
    df = mfm.compare_df_cols('df_trades', 'pnl') \
        .melt(value_name='pnl', var_name='run', ignore_index=False) \
        .reset_index(drop=False)

    sns.scatterplot(x='ts', y='pnl', hue='run', data=df, alpha=0.4, size=0.5)
