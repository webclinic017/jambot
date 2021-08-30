# PCA
# try to idenfity areas with high probability of a large move?
# weight more recent values higher
# need to test constantly retraining model every x hours (24?)
# additive interactions?

# %% - IMPORTS
if True:
    import json
    import os
    import sys
    from datetime import datetime as dt
    from datetime import timedelta as delta
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
    from IPython.display import display
    from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (AdaBoostClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                      Ridge, RidgeClassifier, RidgeCV)
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score, make_scorer, mean_squared_error,
                                 recall_score)
    from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                         TimeSeriesSplit, cross_val_score,
                                         cross_validate, train_test_split)
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                       OrdinalEncoder, PolynomialFeatures,
                                       RobustScaler, StandardScaler)
    from sklearn.svm import SVC
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                              export_graphviz)
    from xgboost import XGBClassifier

    from jambot import charts as ch
    from jambot import data
    from jambot import functions as f
    from jambot import signals as sg
    from jambot import sklearn_utils as sk
    from jambot.database import db
    from jambot.ml import models as md
    from jambot.tradesys import backtest as bt
    from jambot.tradesys.strategies import ml
    from jambot.utils import styles as st

    plt.rcParams.update({'figure.figsize': (12, 5), 'font.size': 14})
    plt.style.use('dark_background')
    pd.set_option('display.max_columns', 200)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blue, red = colors[3], colors[0]

# %% - LOAD DF
if True:
    interval = 15

    kw = dict(
        symbol='XBTUSD',
        # daterange=365 * 6,
        # startdate=dt(2018, 1, 1),
        startdate=dt(2017, 1, 1),
        interval=interval)

    p = Path('df.csv')
    # if p.exists(): p.unlink()

    # read from db or csv
    reload_df = False
    if reload_df or not p.exists() or dt.fromtimestamp(p.stat().st_mtime) < dt.now() + delta(days=-1):
        print('Downloading from db')
        df = db.get_df(**kw)

        df.to_csv(p)
    else:
        df = data.default_df()

    print(f'DateRange: {df.index.min()} - {df.index.max()}')

# %% - ADD SIGNALS

name = 'lgbm'
sm = sg.SignalManager(slope=1, sum=12)

n_periods = 10
p_ema = None  # 6
# Target = sg.TargetMeanEMA
# Target = sg.TargetMean
Target = sg.TargetUpsideDownside
# regression = True
regression = False
# pct_min = 0.01
# pct_min = 0.000000005 # long or short, no neutral
pct_min = 0
target_signal = Target(
    p_ema=p_ema,
    n_periods=n_periods,
    pct_min=pct_min,
    regression=regression)

signals = [
    'EMA',
    'Momentum',
    'Trend',
    'Candle',
    'Volatility',
    'Volume',
    # 'MACD',
    # 'SFP',
    # 'CandlePatterns',
    target_signal
]

# drop last rows which we cant set proper target
# don't need to drop last n_periods rows if positive we aren't fitting on them
df = sm.add_signals(df=df, signals=signals) \
    # .iloc[:-1 * n_periods, :]

if not regression:
    sk.show_prop(df=df)

# %% - FEATURES, MODELMANAGER
if True:
    n_splits = 5
    train_size = 0.9
    # max_train_size = int(df.shape[0] * train_size / n_splits)
    max_train_size = None
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

    if regression:
        scoring = dict(rmse='neg_root_mean_squared_error')
    else:
        scorer = ml.StratScorer(n_smooth=3)
        final_scorer = lambda *args: scorer.score(*args, _type='final')
        max_scorer = lambda *args: scorer.score(*args, _type='max')

        scoring = dict(
            acc='accuracy',
            max=max_scorer,
            final=final_scorer
        )

    cv_args = dict(cv=cv, n_jobs=1, return_train_score=True, scoring=scoring)
    mm = md.make_model_manager(name=name, df=df) \
        .init_cv(scoring=scoring, cv_args=cv_args, scorer=scorer)

    x_train, y_train, x_test, y_test = mm.make_train_test(
        df=df,
        split_date=dt(2021, 1, 1))

# %% - CROSS VALIDATION
#  --%%prun -s cumulative -l 40
LGBM = LGBMRegressor if regression else LGBMClassifier

models = dict(
    # rnd_forest=RandomForestClassifier,
    # xgb=XGBClassifier,
    # ada=AdaBoostClassifier(),
    # log=LogisticRegression(),
    # svc=SVC(gamma=2, C=1, probability=True),
    # nbayes=GaussianNB(),
    # qda=QuadraticDiscriminantAnalysis(),
    # lgbm=LGBM,
    lgbm=LGBM(
        num_leaves=50, n_estimators=50, max_depth=30, boosting_type='dart', learning_rate=0.1)
    # lgbm=LGBM(num_leaves=100, n_estimators=25, max_depth=10, boosting_type='gbdt')
)

# steps = [
#     (1, ('pca', PCA(n_components=20, random_state=0)))]
steps = None

fit_params = dict(fit_params=sk.weighted_fit(name, weights=sg.WeightedPercent(8).get_weight(x_train)))
mm.cross_val(models, steps=steps, extra_cv_args=fit_params)
scorer.show_summary()

# %% - MAXMIN PREDS

if False:
    models_multi = dict(
        lgbm=LGBMRegressor(num_leaves=50, n_estimators=50, max_depth=10, boosting_type='dart'),
        ridge=Ridge()
    )

    models_multi = sk.as_multi_out(models_multi)

    steps = [
        (1, ('pca', PCA(n_components=20, random_state=0)))]
    # steps = None

    mm.cross_val(models_multi, steps=steps)

# %% - OPTIMIZE FEATURES
if False:
    feature_names = 'vol_mfi'
    # params = sm.get_signal_params(feature_names=feature_names)

    windows = [2, 4, 6, 12, 18, 24, 36, 48]
    # windows = [96]
    params = dict(vol_mfi=dict(window=windows))  # , window_slow=windows, cycle=windows))
    # params = dict(trend_cci=dict(constant=[0.05, 0.01, 0.015, 0.02, 0.04, 0.05, 0.06]))
    sk.pretty_dict(params)

    name = 'lgbm'
    mm.cross_val_feature_params(
        signal_manager=sm,
        name=name,
        model=models[name],
        feature_params=params,
        train_size=train_size)

# %% - GRID - LGBMClassifier

best_est = True
if best_est:
    params = dict(
        boosting_type=['dart', 'goss'],
        n_estimators=[25, 50, 100, 200],
        max_depth=[-1, 5, 10, 20, 30, 40],
        num_leaves=[5, 10, 20, 40, 100],
        learning_rate=[0.1, 0.2, 0.5, 0.75]
    )

    # ridge
    # params = dict(
    #     alpha=[0.1, 0.5, 1, 5, 10, 100]
    # )

    grid = mm.search(
        name='lgbm',
        params=params,
        # search_type='grid',
        # refit='acc',
        # refit='rmse',
        refit='final'
    )

# %% - CLASSIFICATION REPORT
# mm.class_rep('lgbm', best_est=best_est)

# %% - RUN STRAT

name = 'lgbm'

# import talib as tb
# df['psar'] = tb.SAR(df.high, df.low, acceleration=0.02, maximum=0.2)
# df['y_pred'] = np.where(df.close > df.psar, -1, 1)
# df_pred = df.copy()

# TODO test iter_predict maxhigh/minlow

if False:
    # retrain every x hours (24 ish) and predict for the test set
    df_pred = mm \
        .add_predict_iter(
            df=df,
            name=name,
            pipe=mm.pipes[name],
            batch_size=24 * 4 * 2,
            min_size=mm.df_train.shape[0],
            max_train_size=None,
            regression=regression)

    df_pred_iter = df_pred.copy()
    # df_pred = df_pred_iter.copy()

else:
    # fit_params = sk.weighted_fit(name, n=mm.df_train.shape[0])
    fit_params = sk.weighted_fit(name, weights=sg.WeightedPercent(8).get_weight(x_train))
    # fit_params = None

    df_pred = mm \
        .add_predict(
            df=df,
            name=name,
            proba=not regression,
            fit_params=fit_params)

n_smooth = 3
rolling_col = 'proba_long' if not regression else 'y_pred'

df_pred = df_pred \
    .pipe(md.add_proba_trade_signal)

strat = ml.Strategy(
    lev=3,
    slippage=0,
    regression=regression,
    market_on_timeout=True,
    # stop_pct=0.025,
)

idx = mm.df_test.index
kw = dict(
    symbol='XBTUSD',
    # daterange=365 * 3,
    # startdate=dt(2020, 8, 1),
    startdate=idx[0],
    # interval=1
)

cols = ['open', 'high', 'low', 'close', 'y_pred', 'proba_long',
        'rolling_proba', 'signal', 'pred_max', 'pred_min', 'target_max',
        'target_min']
bm = bt.BacktestManager(**kw, strat=strat, df=df_pred.pipe(f.clean_cols, cols))
bm.run()
bm.print_final()
wallet = strat.wallet
wallet.plot_balance(logy=True)
trades = strat.trades
t = trades[-1]

# %% - PLOT

periods = 60 * 24
# periods = 400
startdate = dt(2021, 5, 1)
startdate = dt(2021, 1, 1)
# startdate = dt(2020, 3, 1)
startdate = None
split_val = 0.5 if not regression else 0

df_chart = df_pred[df_pred.index >= x_test.index[0]]
# df_chart = df.copy()

traces = [
    # dict(name='buy_pressure', func=ch.bar, color=ch.colors['lightblue']),
    # dict(name='sell_pressure', func=ch.bar, color=ch.colors['lightred']),
    # dict(name='probas', func=ch.probas),
    # dict(name='target'),
    dict(name=rolling_col, func=ch.split_trace, split_val=split_val),
    dict(name='rolling_proba', func=ch.split_trace, split_val=split_val),
    # dict(name='mnt_rsi_2', func=ch.scatter),
    # dict(name='volume', func=ch.bar),
]

df_balance = strat.wallet.df_balance
# df_balance = None

if not df_balance is None:
    traces.append(dict(name='balance', func=ch.scatter, color='#91ffff', stepped=True))

    df_chart = df_chart \
        .merge(right=df_balance, how='left', left_index=True, right_index=True) \
        .assign(balance=lambda x: x.balance.fillna(method='ffill'))


# Add trade_side for trade entry indicators in chart
df_trades = strat.df_trades()

if not df_trades is None:
    traces.append(dict(name='trades', func=ch.trades, row=1))

    rename_cols = dict(
        side='trade_side',
        pnl='trade_pnl',
        entry='trade_entry',
        exit='trade_exit')

    df_trades = df_trades \
        .set_index('ts') \
        .rename(columns=rename_cols)

    df_chart = df_chart.pipe(f.left_merge, df_trades)

# enumerate row numbers
# traces = [{**m, **dict(row=m.get('row', i+2))} for i, m in enumerate(traces)]
# traces = None

fig = ch.chart(
    df=df_chart,
    periods=periods,
    last=True,
    startdate=startdate,
    df_balance=df_balance,
    traces=traces,
    regression=regression)
fig.show()


# %% - DENSITY PLOTS
# # %% time
data = mm.ct.fit_transform(x_train)

expr = 'sfp'
df_trans = sk.df_transformed(data=data, ct=mm.ct) \
    .assign(target=y_train) \
    .pipe(f.select_cols, expr=expr, include='target')

melt_cols = df_trans.columns[df_trans.columns != 'target']
df_melted = df_trans \
    .rename_axis('index') \
    .reset_index() \
    .melt(id_vars=['index', 'target'], value_vars=melt_cols)

fg = sns.FacetGrid(
    df_melted,
    col='variable',
    col_wrap=6,
    hue='target',
    palette=[blue, red],
    sharex=False,
    sharey=False,
    height=2.5)

fg \
    .map(
        sns.kdeplot,
        'value',
        shade=True,
        label='Data')\
    .add_legend()\
    .set_titles('{col_name}')\
    # .set_axis_labels('Feature Value', 'Density')

plt.show()

# %% - SHAP PLOT
mm.shap_plot(name=name)

# %%
explainer, shap_values, x_sample, x_enc = sk \
    .shap_explainer_values(x_test, y_test, mm.ct, mm.models['lgbm'])

# %%
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], x_enc.iloc[0, :])

# %% - RIDGE
models = dict(
    lgbm_rfecv=mm.models['lgbm'])

extra_steps = (1, ('rfecv', RFE(Ridge())))  # must insert rfecv BEFORE other model
mm.cross_val(models, steps=extra_steps)
scorer.show_summary()

# %% - RFECV SHOW FEATURES
data = ct.fit_transform(x_train)
df_trans = sk.df_transformed(data=data, ct=ct).describe().T

pipe_rfe = mm.pipes['lgbm_rfecv']
pipe_rfe.fit(x_train, y_train)  # need fit pipe again outside of cross_validate
rfecv = pipe_rfe.named_steps['rfecv']
print('n_features: ', rfecv.n_features_)

pd.DataFrame(
    data=rfecv.support_,
    columns=['Included'],
    index=df_trans.index) \
    .style \
    .apply(st.highlight_val, subset=['Included'], m={True: (ch.colors['lightblue'], 'black')})

# %%
# POLY
models = dict(
    lgbm_poly=mm.models['lgbm'])

# note - these are inserted in reverse order
extra_steps = [
    # (1, ('rfecv', RFECV(Ridge()))),
    (1, ('poly', PolynomialFeatures(degree=2)))]

mm.cross_val(models, steps=extra_steps)

# %%
pipe_rfe = mm.pipes['lgbm_poly']
pipe_rfe.fit(x_train, y_train)
rfecv = pipe_rfe.named_steps['rfecv']
rfecv.n_features_
