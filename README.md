# jambot
*Code only as a portfolio example, not yet ready for public use.

In production, this app is hosted as an Azure Functions app running on a timer trigger, performing certain trade functions/checks/notifications on 5min and 1hr intervals.

>## Project
>**Functions** - Helper functions.
>
>**[JambotClasses](Project/JambotClasses.py)** - Main classes for backtesting strategies.
>
>**Launch** - Control module for local development, testing, backtesting strategies, multiprocess optimization etc. Used in VS Code with a Python Interactive (jupyter) environment.
>
>**[LiveTrading](Project/LiveTrading.py)** - Interaction with Bitmex for executing trades, and Google sheets for live dashboard of current status.
>run_toploop() is main control function, runs every 1 hour

>## Azure Functions App - Triggers
>**FiveMin** - Checks for filled orders and specific candle patterns, sends Discord alert messages.
>
>**OneHour** - Runs run_toploop() every hour.
>
>**HttpTrigger** - Respond to http requests from user dashboard to perform functions, place orders, reset strategy, refresh balance etc

## Example
``` py
from datetime import datetime as dt

from jambot import (
    functions as f,
    livetrading as live,
    backtest as bt,
    charts as ch)
from jambot.database import db

symbol = 'XBTUSD'
daterange = 365
startdate = dt(2019, 1, 1)
interval = 1

# Load OHLC data from Azure-hosted SQL server 
df = db.get_df(symbol=symbol, startdate=startdate, daterange=daterange, interval=interval)

# Init TrendRev strategy
speed = (16, 6)
norm = (0.004, 0.024)
strat = bt.Strat_TrendRev(speed=speed, norm=norm)
strat.slippage = 0
strat.stop_pct = -0.03
strat.timeout = 40

# BacktestManager strategy and plot balance
bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=[strat], df=df)
bm.decide_full()
bm.print_final()
bm.account.plot_balance(logy=True)
```
![PlotBalance](docs/pics/PlotBalance.png)

```
strat.print_trades(last=15)
```
![PrintTrades](docs/pics/PrintTrades.png)

Use plotly to create interactive chart with order fill prices. Very useful to look through backtested trades and see how the strategy performed in each case.
```
t = strat.trades[8]
ch.chart_orders(t)
```
![PlotTrade](docs/pics/PlotTrade.png)