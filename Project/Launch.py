#%% IMPORTS
if True:
	import cProfile
	import json
	import os
	import sys
	from datetime import datetime as date
	from datetime import timedelta as delta
	from pathlib import Path
	from time import time
	from timeit import Timer

	from IPython.display import display
	import numpy as np
	import pandas as pd
	import plotly.offline as py
	from joblib import Parallel, delayed

	import Functions as f
	import LiveTrading as live
	import JambotClasses as c

	startdate, daterange = date(2017, 1, 14), 365
	symbol = 'XBTUSD'

	# t = Timer(lambda: fldr.readsingle(p))
	# print(t.timeit(number=10))

#%% - Save db to csv
if True:
	startdate, daterange = date(2017, 1, 1), 365*4
	df = f.getDataFrame(symbol=None, startdate=startdate, daterange=daterange)
	df.to_csv('df.csv')

#%% - BACKTEST
if True:
	start = time()
	symbol = 'XBTUSD'
	daterange = 365 * 3 
	# startdate = date(2018, 1, 1)
	# startdate = date(2019, 1, 1)
	startdate = date(2019, 7, 1)
	interval = 1

	df = f.getDataFrame(symbol=symbol, startdate=startdate, daterange=daterange, interval=interval)

	# TREND_REV
	speed = (16, 6)
	norm = (0.004, 0.024)
	strat = c.Strat_TrendRev(speed=speed, norm=norm)
	strat.slippage = 0
	strat.stoppercent = -0.03
	strat.timeout = 40

	# TREND
	# speed = (25, 18)
	# strat = Strat_Trend(speed=speed)
	# strat.slippage = 0.002
	# strat.enteroffset = 0.015

	# TREND_CLOSE
	# speed = (25, 18)
	# if interval == 15:
	# 	speed = tuple(x * 4 for x in speed)
	# strat = Strat_TrendClose(speed=speed)

	# SFP
	# strat = Strat_SFP()

	sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strat, df=df, partial=False)
	sym.decidefull()
	f.printTime(start)
	sym.printfinal()
	sym.account.plotbalance(logy=True)
	trades = strat.trades
	t = trades[-1]

#%%
# BACKTEST ALL - PARALLEL
if True:
	import seaborn as sns

	dfsym = pd.read_csv(os.path.join(f.curdir(), 'symbols.csv')).query('enabled==True')
	startdate, daterange = date(2019, 1, 1), 720
	dfall = f.getDataFrame(startdate=startdate, daterange=daterange)

	start = time()
	
	# TREND_REV
	strattype = 'trendrev'
	speed = (16, 6)
	norm = (0.004, 0.024)

	# TREND
	# strattype = 'trend'
	# speed = (25, 18)
	# norm = None
	
	syms = Parallel(n_jobs=-1)(delayed(f.run_single)(strattype, startdate, dfall, speed[0], speed[1], row, norm) for row in dfsym.itertuples())

	results = [sym.result() for sym in syms]
	cmap = sns.diverging_palette(10, 240, sep=80, n=7, as_cmap=True)
	df = pd.concat(results).reset_index(drop=True)
	style = df.style.format({'Min': '{:.3f}',
				'Max': '{:.3f}',
				'Final': '{:.3f}',
				'Drawdown': '{:.2%}'}) \
				.hide_index() \
				.background_gradient(cmap=cmap)	

	display(style)

	# for sym in syms:
	# 	sym.account.plotbalance(logy=True, title=sym.symbol)

	f.printTime(start)

#%%
# PARALLEL OPTIMIZATION
if True:
	dfsym = pd.read_csv(Path(f.curdir()) / 'symbols.csv')
	startdate, daterange = date(2019, 1, 1), 365 * 3
	# dfall = getDataFrame(startdate=startvalue(startdate), enddate=enddate(startdate, daterange))

	symbol = 'XBTUSD'
	# update csv from database before running!!
	dfall = f.readcsv(startdate, daterange, symbol=symbol)
	start = time()

	titles = ('against', 'wth')
	# titles = ('uppernorm', 'lowernorm')
	# titles = ('tpagainst', 'tpwith')

	for row in dfsym[dfsym.symbol==symbol].itertuples():		
		if True:
			# TREND_REV
			strattype = 'trendrev'
			norm = (0.004, 0.024)
			syms = Parallel(n_jobs=-1)(delayed(f.run_single)(strattype, startdate, dfall, speed0, speed1, row, norm) for speed0 in range(6, 23, 1) for speed1 in range(6, 19, 1))
		
		if False:
			# TREND
			speed = (1,1)
			mr = False
			opposite = False
			listout = Parallel(n_jobs=-1)(delayed(f.runtrend)(symbol, startdate, mr, df, against, wth, row, titles) for against in range(8, 44, 2) for wth in range(8, 44, 2))
		
		if False:
			# CHOP
			# uppernormal/lowernormal
			# speed = (row.against2, row.with2)
			speed = (36, 26)
			speedtp = (26, 16)
			norm = [12,1]
			
			# speed
			# , backend='multiprocessing'
			listout = Parallel(n_jobs=-1)(delayed(f.runchop)(symbol, startdate, df, speed[0], speed[1], tpagainst, tpwith, norm[0], norm[1], row, titles) for tpagainst in range(14, 45, 2) for tpwith in range(14, 45, 2))

			# norm
			# listout = Parallel(n_jobs=-1)(delayed(runchop)(symbol, startdate, df, speed[0], speed[1], speedtp[0], speedtp[1], lowernorm, uppernorm, row, titles) for lowernorm in range(1, 25) for uppernorm in range(1, 25))

		f.printTime(start)
		
		dfResults = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])
		for i, sym in enumerate(syms):
			strat = sym.strats[0]
			a = sym.account
			speed = strat.speed

			dfResults.loc[i] = [speed[0], speed[1], round(a.min,3), round(a.max,3), round(a.balance,3), strat.tradecount()]

		size = 15
		dims = (len(dfResults[titles[0]].unique()), len(dfResults[titles[1]].unique()))
		dims = (min(dims), max(dims))
		d = (int(np.interp(dims[1], dims, (size * (dims[0] / dims[1]), size))),
			int(np.interp(dims[0], dims, (size * (dims[0] / dims[1]), size))))
		
		f.heatmap(dfResults, [titles[0], titles[1], 'max'], title='{} - {} - {} - Max'.format(symbol, speed, startdate), dims=d)
		f.heatmap(dfResults, [titles[0], titles[1], 'final'], title='{} - {} - {} - Final'.format(symbol, speed, startdate), dims=d)


#%% - MATPLOT
if True:
	import matplotlib.pyplot as plt
	startdate, daterange = date(2019, 5, 1), 365

	df = f.readcsv(startdate, daterange)
	fig, ax1 = plt.subplots()

	df.plot(kind='line', x='CloseTime', y='Close', ax=ax1)
	ax2=ax1.twinx()
	df.plot(kind='line', x='CloseTime', y='spread', linewidth=0.5, color='salmon', ax=ax2)
	df.plot(kind='line', x='CloseTime', y='emaVty', linewidth=0.5, color='cyan', ax=ax2)
	plt.show()

#%% - PLOTLY
if True:
	import plotly.graph_objs as go
	import plotly.offline as py
	
	symbol = 'XBTUSD'
	startdate, daterange = date(2019, 11, 1), 720
	dfall = f.readcsv(startdate, daterange)
	df = f.filterdf(dfall=dfall, symbol=symbol)
	# df = getDataFrame(symbol=symbol, startdate=startvalue(startdate), enddate=enddate(startdate, daterange))
	norm=(0.04, 0.24)
	vty = c.Signal_Volatility(df=df, norm=norm)

	trace1 = go.Scatter(x=df['CloseTime'], y=df['spread'],line=dict(color='salmon', width=1))
	
	trace2 = go.Scatter(x=df['CloseTime'], y=df['smavty'], line=dict(color='#f5f55f', width=1))
	trace3 = go.Scatter(x=df['CloseTime'], y=df['norm_sma'], line=dict(color='#f5f55f', width=1, dash='dash'))
	
	trace4 = go.Scatter(x=df['CloseTime'], y=df['emavty'],line=dict(color='cyan', width=1))
	trace5 = go.Scatter(x=df['CloseTime'], y=df['norm_ema'],line=dict(color='cyan', width=1,dash='dash'))
	
	candle = go.Candlestick(
		x=df['CloseTime'],
		open=df['Open'],
		high=df['High'],
		low=df['Low'],
		close=df['Close'],
		increasing=dict(line=dict(color='white')),
		decreasing=dict(line=dict(color='grey')),
		line=dict(width=1),
		yaxis='y2')
	data = [trace1, trace2, trace3, trace4, trace5, candle]
	layout = go.Layout(
		height=800,
		width=800,
		paper_bgcolor='#000B15',
		plot_bgcolor='#000B15',
		font_color='white',
		showlegend=False,
		xaxis=dict(showgrid=False,
			autorange=True),
		yaxis=dict(showgrid=False,
			autorange=True,
			fixedrange=False),
		yaxis2=dict(side='right',
			overlaying='y',
			showgrid=False,
			autorange=True,
			fixedrange=False))
	fig = go.Figure(data=data, layout=layout)
	fig.show(config=dict(scrollZoom=True))


#%% - PROFILE
import pstats

filename = 'profile_stats.stats'
symbol = 'XBTUSD'
strattype = 'trendrev'
startdate, daterange = date(2019, 1, 1), 365
df = f.readcsv(startdate, daterange, symbol=symbol)
speed = (16, 6)
norm = (0.004, 0.024)

# cProfile.run('run_single(symbol=symbol,\
# 						strattype=strattype,\
# 						startdate=startdate,\
# 						dfall=df,\
# 						speed0=speed[0],\
# 						speed1=speed[1],\
# 						norm=norm)',
# 						filename = filename)		
						
cProfile.run('testdf(df)', filename = filename)

# cProfile.run('run_parallel()', filename=filename)
stats = pstats.Stats(filename)
stats.strip_dirs().sort_stats('cumulative').print_stats(30)

#%%
fig = f.chart(df, symbol=symbol)
py.iplot(fig)

#%% - SFP Test
if True:
	symbol = 'XBTUSD'
	daterange = 365 * 3
	startdate = date(2019, 7, 1)

	df = f.getDataFrame(symbol=symbol, startdate=startdate, daterange=daterange)

	sfp = c.Strat_SFP()
	sfp.init(df=df)

# Test trigger azure function running on localhost
# curl --request POST -H "Content-Type:application/json" --data '{"input":""}' http://localhost:7071/admin/functions/FiveMin