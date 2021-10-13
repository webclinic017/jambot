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
    from jambot import config as cf
    from jambot import data
    from jambot import functions as f
    from jambot import getlog
    from jambot import signals as sg
    from jambot import sklearn_utils as sk
    from jambot.database import db
    from jambot.exchanges.bitmex import Bitmex
    from jambot.ml import models as md
    from jambot.tradesys import backtest as bt
    from jambot.tradesys.strategies import ml as ml
    from jambot.utils import styles as st

    log = getlog(__name__)

    plt.rcParams |= {'figure.figsize': (12, 5), 'font.size': 14, 'lines.linewidth': 1.0}
    plt.style.use('dark_background')
    pd.set_option('display.max_columns', 200)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blue, red = colors[3], colors[0]

# %% - LOAD DF
if True:
    interval = 15
    p = Path('df.csv')
    # if p.exists(): p.unlink()

    # read from db or csv
    reload_df = False
    if reload_df or not p.exists() or dt.fromtimestamp(p.stat().st_mtime) < dt.now() + delta(days=-1):
        log.info('Downloading from db')
        df = db.get_df(
            symbol='XBTUSD',
            startdate=dt(2017, 1, 1),
            interval=interval)

        df.to_csv(p)
    else:
        df = data.default_df()

    log.info(f'DateRange: {df.index.min()} - {df.index.max()}')

# %% - ADD SIGNALS
# --%%prun -s cumulative -l 40
# --%%time

if True:
    name = 'lgbm'
    cfg = md.model_cfg(name)

    slope = [1, 4, 8, 16, 32, 64]
    _sum = [12, 24, 96]
    # slope = 1
    # _sum = 12
    sm = sg.SignalManager(slope=slope, sum=_sum)

    n_periods = cfg['target_kw']['n_periods']
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
        # 'DateTime',
        # 'MACD',
        # 'SFP',
        # 'CandlePatterns',
        target_signal
    ]

    # drop last rows which we cant set proper target
    # don't need to drop last n_periods rows if positive we aren't fitting on them
    use_important = True
    df = sm.add_signals(df=df, signals=signals, use_important=use_important) \
        # .iloc[:-1 * n_periods, :]
    if not use_important:
        df_all = df.copy()

    if not regression:
        sk.show_prop(df=df)

# %% - FEATURES, MODELMANAGER
if True:
    n_splits = 5
    train_size = 0.9
    # max_train_size = int(df.shape[0] * train_size / n_splits)
    # max_train_size = 20_000
    max_train_size = None
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

    weights = sg.WeightedPercentMaxMin(
        n_periods=cfg['n_periods_weighted'],
        weight_linear=False) \
        .get_weight(df)

    if regression:
        scoring = dict(rmse='neg_root_mean_squared_error')
    else:
        scorer = ml.StratScorer()

        scoring = dict(
            acc='accuracy',
            wt=make_scorer(sk.weighted_score, weights=weights),
            max=lambda *args: scorer.score(*args, _type='max'),
            final=lambda *args: scorer.score(*args, _type='final')
        )

    cv_args = dict(cv=cv, n_jobs=-1, return_train_score=True, scoring=scoring)
    mm = md.make_model_manager(name=name, df=df, use_important=use_important)

    x_train, y_train, x_test, y_test = mm.make_train_test(
        df=df,
        split_date=dt(2021, 1, 1))

    cv_args |= dict(
        fit_params=sk.weighted_fit(name, weights=weights[x_train.index]),
        return_estimator=True)

    mm.init_cv(scoring=scoring, cv_args=cv_args, scorer=scorer)

    models = dict(
        lgbm=LGBMClassifier(
            num_leaves=50, n_estimators=50, max_depth=30, boosting_type='dart', learning_rate=0.1))

    mm.init_models(models)

    log.info(f'num_feats: {len(mm.ct.transformers[1][2])}')

# %% - CROSS VALIDATION

# --%%time
#  --%%prun -s cumulative -l 40

# steps = [
#     (1, ('pca', PCA(n_components=20, random_state=0)))]
steps = None
mm.cross_val(models, steps=steps)
res_dfs = [m.cv_data['df_result'] for m in mm.cv_data[name]]
scorer.show_summary(dfs=res_dfs, scores=mm.scores[name])

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

# TODO test iter_predict maxhigh/minlow

if True:
    # fit_params = sk.weighted_fit(name, n=mm.df_train.shape[0])
    fit_params = sk.weighted_fit(
        name=name,
        weights=sg.WeightedPercentMaxMin(8, weight_linear=True).get_weight(x_train))
    # fit_params = None

    df_pred = mm \
        .add_predict(
            df=df,
            name=name,
            proba=not regression,
            fit_params=fit_params)

else:
    # retrain every x hours (24 ish) and predict for the test set
    df_pred = mm \
        .add_predict_iter(
            df=df,
            name=name,
            batch_size=24 * 4 * 4,
            min_size=mm.df_train.shape[0],
            max_train_size=None,
            regression=regression)

    df_pred_iter = df_pred.copy()
    # df_pred = df_pred_iter.copy()

df_pred = df_pred \
    .pipe(md.add_proba_trade_signal)

strat = ml.make_strat()

cols = ['open', 'high', 'low', 'close', 'y_pred', 'proba_long',
        'rolling_proba', 'signal', 'pred_max', 'pred_min', 'target_max',
        'target_min']
bm = bt.BacktestManager(
    startdate=mm.df_test.index[0],
    strat=strat,
    df=df_pred.pipe(f.clean_cols, cols)).run(prnt=True)

# %% - PLOT

periods = 60 * 24 * 4
startdate = dt(2021, 5, 1)
startdate = dt(2021, 8, 1)
startdate = None

df_balance = strat.wallet.df_balance
df_trades = strat.df_trades()

# cv data
# i = 3
# m = mm.cv_data['lgbm'][i].cv_data
# startdate = m['startdate']
# startdate = dt(2020, 3, 1)
# df_balance = m['df_balance']
# df_trades = m['df_trades']

ch.plot_strat_results(
    df=df_pred.pipe(f.clean_cols, cols + ['target']),
    df_balance=df_balance,
    df_trades=df_trades,
    startdate=startdate,
    periods=periods)

# %% - INIT SHAP MANAGER
x_shap = df.drop(columns=['target'])
y_shap = df.target
# x_shap = x_test
# y_shap = y_test
# x_shap = x_train
# y_shap = y_train
sm = sk.ShapManager(x=x_shap, y=y_shap, ct=mm.ct, model=mm.models['lgbm'], n_sample=10_000)

# %% - SHAP PLOT
sm.plot(plot_type='violin')

# %%
sm.force_plot(sample_n=0)

# %%
res = sm.shap_n_important(n=60, save=True, upload=False, as_list=True)
cols = res['most']
cols


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


# %% - RIDGE
models = dict(
    lgbm_rfecv=mm.models['lgbm'])

extra_steps = (1, ('rfecv', RFE(
    estimator=Ridge(),
    n_features_to_select=50)))  # must insert rfecv BEFORE other model
mm.cross_val(models, steps=extra_steps)
scorer.show_summary()

# %% - RFECV SHOW FEATURES
ct = mm.ct
data = ct.fit_transform(x_train)
df_trans = sk.df_transformed(data=data, ct=ct).describe().T

model_rfe = RFE(estimator=Ridge(), n_features_to_select=50, step=3)
pipe = mm.make_pipe(name='rfe', model=model_rfe)

fit_params = sk.weighted_fit(
    name='rfe',
    weights=sg.WeightedPercentMaxMin(8, weight_linear=False).get_weight(x_train))
pipe.fit(x_train, y_train)

# pipe_rfe = mm.pipes['lgbm_rfecv']
# pipe_rfe.fit(x_train, y_train)  # need fit pipe again outside of cross_validate
# rfecv = pipe_rfe.named_steps['rfecv']
log.info(f'n_features: {model_rfe.n_features_}')

df_rfe = pd.DataFrame(
    data=model_rfe.support_,
    columns=['included'],
    index=df_trans.index)

style_rfe = df_rfe \
    .style \
    .apply(st.highlight_val, subset=['included'], m={True: (ch.colors['lightblue'], 'black')})

style_rfe

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

# %% ARCHIVE
# rnd_forest=RandomForestClassifier,
# xgb=XGBClassifier,
# ada=AdaBoostClassifier(),
# log=LogisticRegression(),
# svc=SVC(gamma=2, C=1, probability=True),
# nbayes=GaussianNB(),
# qda=QuadraticDiscriminantAnalysis(),
# lgbm=LGBM,
