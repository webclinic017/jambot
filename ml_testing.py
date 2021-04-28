# PCA
# try to idenfity areas with high probability of a large move?
# weight more recent values higher
# try min_agree cumulative % threshold instead of number
# need to test constantly retraining model every x hours (24?)
# additive interactions?

#%% - IMPORTS
if True:
    import json
    import os
    import sys
    from datetime import datetime as dt
    from datetime import timedelta as delta
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    plt.rcParams.update({"font.size": 14})
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
    from IPython.display import display
    from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

    from sklearn.compose import ColumnTransformer, make_column_transformer
    # Classifiers
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                                RandomForestClassifier)
    # other
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.impute import SimpleImputer
    # classifiers / models
    from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge,
                                    RidgeCV)
    from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                                make_scorer, recall_score, mean_squared_error)
    from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                        cross_val_score, cross_validate,
                                        train_test_split, TimeSeriesSplit)
    from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
    from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                    PolynomialFeatures, RobustScaler,
                                    StandardScaler)
    from sklearn.svm import SVC
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                            export_graphviz)
    from xgboost import XGBClassifier

    from jambot import backtest as bt
    from jambot import charts as ch
    from jambot import functions as f
    from jambot import livetrading as live
    from jambot import optimization as op
    from jambot import signals as sg
    from jambot import sklearn_helper_funcs as sf
    from jambot.strategies import ml
    from jambot.database import db
    from jambot.strategies import trendrev

    # from mlxtend.feature_selection import SequentialFeatureSelector

#%% - LOAD DF
if True:
    kw = dict(
        symbol='XBTUSD',
        daterange=365 * 5,
        startdate=dt(2018, 1, 1),
        # startdate=dt(2017, 1, 1),
        interval=1)

    p = Path('df.csv')
    # if p.exists(): p.unlink()

    # read from db or csv
    if not p.exists() or dt.fromtimestamp(p.stat().st_mtime) < dt.now() + delta(days=-1):
        df = db.get_dataframe(**kw) \
            .drop(columns=['Timestamp', 'Symbol'])
            
        df.to_csv(p)
    else:
        df = pd.read_csv(p, parse_dates=['Timestamp'], index_col='Timestamp')

    print(f'DateRange: {df.index.min()} - {df.index.max()}')

# %% - ADD SIGNALS

sm = sg.SignalManager()

n_periods = 6
p_ema = None # 6
# Target = sg.TargetMeanEMA
Target = sg.TargetMean
# regression = True
regression = False
# pct_min = 0.01
# pct_min = 0.000000005 # long or short, no neutral
pct_min = 0
target_signal = Target(p_ema=p_ema, n_periods=n_periods, pct_min=pct_min, regression=regression)
# sm_target = sg.SignalManager()
# df = df.pipe(sm_target.add_signals, signals=[target_signal])

signals = [
    'EMA',
    'Momentum',
    'Trend',
    'Candle',
    'EMASlope',
    'Volatility',
    'Volume',
    # 'MACD',
    'SFP',
    # 'CandlePatterns',
    target_signal
    ]

# drop last rows which we cant set proper target
df = sm.add_signals(df=df, signals=signals) \
    .iloc[:-1 * n_periods, :] \
    .dropna()

if not regression:
    sf.show_prop(df=df)

#%% - FEATURES
if True:
    cols_ohlcv = ['Open', 'High', 'Low', 'Close', 'VolBTC']
    drop_feats = cols_ohlcv + ['ema10', 'ema50', 'ema200', 'pxhigh', 'pxlow']

    features = dict(
        # signalmanager=cols_ohlcv,
        target=['target'],
        drop=drop_feats,
        passthrough=['ema_trend', 'ema_slope', 'cdl_side', 'macd_trend']
        )

    features['passthrough'] = features['passthrough'] + [col for col in df.columns if any(item in col for item in ('sfp', 'prevhigh', 'prevlow'))]
    features['numeric'] = [col for col in df.columns if not any(col in lst for lst in features.values())]

    # remove any cols not in df
    features = {name: [col for col in cols if col in df.columns] for name, cols in features.items()}
    
    encoders = dict(
        drop='drop',
        passthrough='passthrough',
        numeric=MinMaxScaler(feature_range=(0.5, 1)),
        )

    # cv = 5
    n_splits = 3
    train_size = 0.9
    max_train_size = df.shape[0] * train_size / n_splits
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=int(max_train_size))

    if not regression:
        scoring = dict(acc='accuracy')
    else:
        scoring = dict(rmse='neg_root_mean_squared_error')

    cv_args = dict(cv=cv, n_jobs=-2, return_train_score=True, scoring=scoring)
    mm = sf.ModelManager(scoring=scoring, cv_args=cv_args)
    ct = mm.make_column_transformer(features=features, encoders=encoders)

    x_train, y_train, x_test, y_test = mm.make_train_test(
        df=df,
        target=features['target'],
        train_size=train_size,
        shuffle=False)

#%% - SHOW COLUMN TRANSFORMS
# mm.show_ct()

#%% - MODELS
LGBM = LGBMRegressor if regression else LGBMClassifier

models = dict(
    # rnd_forest=RandomForestClassifier,
    # xgb=XGBClassifier,
    # ada=AdaBoostClassifier(),
    lgbm=LGBM(num_leaves=50, n_estimators=50, max_depth=10, boosting_type='dart')
    )
    
mm.cross_val(models)

#%% - OPTIMIZE FEATURES
if False:
    feature_names = 'mnt_awesome'
    params = sm.get_signal_params(feature_names=feature_names)

    windows = [2, 6, 12, 18, 24, 36, 48]
    windows = [96]
    params = dict(trend_trix=dict(window=windows)) #, window_slow=windows, cycle=windows))
    # params = dict(trend_cci=dict(constant=[0.05, 0.01, 0.015, 0.02, 0.04, 0.05, 0.06]))
    sf.pretty_dict(params)

    name='lgbm'
    mm.cross_val_feature_params(signal_manager=sm, name=name, model=models[name], feature_params=params)

#%% - GRID - LGBMClassifier

best_est = False
if best_est:
    params = dict(
        boosting_type=['gbdt', 'dart', 'goss', 'rf'],
        n_estimators=[25, 50, 100],
        max_depth=[-1, 5, 10, 20, 30],
        num_leaves=[5, 10, 20, 40, 100],
        )

    grid = mm.search(
        name='lgbm',
        params=params,
        search_type='grid',
        refit='acc')

#%% - CLASSIFICATION REPORT
# mm.class_rep('lgbm', best_est=best_est)

#%% GRID - AdaBoostClassifier
run_ada = False
if run_ada:
    params = dict(
        algorithm=['SAMME', 'SAMME.R'],
        n_estimators=[50, 100, 200, 400],
        learning_rate=[0.0125, 0.25, 0.5, 1, 2])

    grid = mm.search(
        name='ada',
        params=params,
        search_type='random',
        refit='acc')

#%% - ADD PREDICT
name = 'lgbm'
model = models[name]
# df2 = df[df.index > '2020-06-01'].copy()

n_smooth = 6
rolling_col = 'proba_long' if not regression else 'y_pred'

# df_pred = mm.add_predict_iter(df=df, name=name, model=models[name], batch_size=24, max_train_size=None, regression=regression) \
# df_pred = df_pred \
#     .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba')
    # .assign(rolling_proba=lambda x: x[rolling_col].rolling(n_smooth).mean())
# df_pred_iter = df_pred.copy()
# df_pred = df_pred_iter.copy()

#%% - RUN STRAT
# name = 'lgbm_poly'
df_pred = mm.add_predict(df=df, name=name, best_est=False, proba=not regression) \
    .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba')
    # .assign(rolling_proba=lambda x: x[rolling_col].rolling(n_smooth).mean())

strat = ml.Strategy(
    min_agree=10,
    lev=3,
    min_proba=0.5,
    slippage=0,
    regression=regression,
    # stoppercent=-0.025,
    )
idx = mm.df_test.index

kw = dict(
    symbol='XBTUSD',
    # daterange=365 * 3,
    # startdate=dt(2020, 8, 1),
    startdate=idx[0],
    # interval=1
    )

sym = bt.Backtest(**kw, strats=strat, df=df_pred)
sym.decide_full()
# f.print_time(start)
sym.print_final()
sym.account.plot_balance(logy=True)
trades = strat.trades
t = trades[-1]

#%% - PLOT

periods = 720
# periods = 400
# startdate = dt(2020, 8, 1)
startdate = None
split_val = 0.5 if not regression else 0

df_chart = df_pred[df_pred.index >= x_test.index[0]]
# df_chart = df.copy()

traces = [
    # dict(name='buy_pressure', func=ch.bar, color=ch.colors['lightblue']),
    # dict(name='sell_pressure', func=ch.bar, color=ch.colors['lightred']),
    # dict(name='probas', func=ch.probas),
    dict(name=rolling_col, func=ch.split_trace, split_val=split_val),
    dict(name='rolling_proba', func=ch.split_trace, split_val=split_val),
    # dict(name='vol_relative', func=ch.scatter),
    # dict(name='VolBTC', func=ch.bar),
]

df_balance = strat.a.df_balance
# df_balance = None

if not df_balance is None:
    traces.append(dict(name='balance', func=ch.scatter, color='#91ffff', stepped=True))
    
    df_chart = df_chart.merge(right=df_balance, how='left', left_index=True, right_index=True) \
        .assign(balance=lambda x: x.balance.fillna(method='ffill'))


# Add trade_side for trade entry indicators in chart
df_trades = strat.result()
# df_trades = None

if not df_trades is None:
    traces.append(dict(name='trades', func=ch.trades, row=1))

    rename_cols = dict(
        Sts='trade_side',
        PnlAcct='trade_pnl',
        Entry='trade_entry',
        Exit='trade_exit')

    df_trades = df_trades.set_index('Timestamp') \
        .rename(columns=rename_cols) \
        [rename_cols.values()]

    df_chart = df_chart.merge(right=df_trades, how='left', left_index=True, right_index=True)
        
# enumerate row numbers
# traces = [{**m, **dict(row=m.get('row', i+2))} for i, m in enumerate(traces)]
# traces = None

fig = ch.chart(
    df=df_chart,
    periods=periods,
    last=True,
    startdate=startdate,
    df_balance=strat.a.df_balance,
    traces=traces,
    regression=regression)
fig.show()

#%% - SHAP PLOT
mm.shap_plot(name=name)

#%% - RIDGE
models = dict(
    lgbm_rfecv=mm.models['lgbm'])

extra_steps = (1, ('rfecv', RFECV(Ridge()))) # must insert rfecv BEFORE other model
mm.cross_val(models, steps=extra_steps)

#%% - RFECV SHOW FEATURES
data = ct.fit_transform(x_train)
df_trans = sf.df_transformed(data=data, ct=ct).describe().T

pipe_rfe = mm.pipes['lgbm_rfecv']
pipe_rfe.fit(x_train, y_train) # need fit pipe again outside of cross_validate
rfecv = pipe_rfe.named_steps['rfecv']
rfecv.n_features_

pd.DataFrame(data=rfecv.support_, columns=['Included'], index=df_trans.index) \
    .style \
    .apply(sf.highlight_val, subset=['Included'], m={True: (ch.colors['lightblue'], 'black')})

#%%
# POLY
models = dict(
    lgbm_poly=mm.models['lgbm'])

# note - these are inserted in reverse order
extra_steps = [
    # (1, ('rfecv', RFECV(Ridge()))),
    (1, ('poly', PolynomialFeatures(degree=2)))]

mm.cross_val(models, steps=extra_steps)

#%%
pipe_rfe = mm.pipes['lgbm_poly']
pipe_rfe.fit(x_train, y_train)
rfecv = pipe_rfe.named_steps['rfecv']
rfecv.n_features_