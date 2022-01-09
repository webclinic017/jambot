# TODO secondary model for "trade or no trade?"

# %% - IMPORTS

if True:
    from datetime import datetime as dt
    from datetime import timedelta as delta
    from itertools import product
    from pathlib import Path

    import lightgbm as lgb
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (AdaBoostClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                      Ridge, RidgeClassifier, RidgeCV)
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score, make_scorer, mean_squared_error,
                                 recall_score)
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                              export_graphviz)

    from jambot import charts as ch
    from jambot import config as cf
    from jambot import data
    from jambot import database as dtb
    from jambot import functions as f
    from jambot import getlog
    from jambot import livetrading as live
    from jambot import signals as sg
    from jambot import sklearn_utils as sk
    from jambot.livetrading import ExchangeManager
    from jambot.ml import models as md
    from jambot.ml.classifiers import LGBMClsLog
    from jambot.tables import Tickers
    from jambot.tradesys import backtest as bt
    from jambot.tradesys.strategies import ml as ml
    from jambot.utils import skopt as sko
    from jambot.utils import styles as st
    from jambot.utils.mlflow import MlflowManager
    from jambot.weights import WeightsManager
    from jgutils import pandas_utils as pu

    log = getlog(__name__)

    mfm = MlflowManager()

    plt.rcParams |= {'figure.figsize': (12, 5), 'font.size': 14, 'lines.linewidth': 1.0}
    plt.style.use('dark_background')
    pd.set_option('display.max_columns', 200)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blue, red = colors[3], colors[0]

# %% - LOAD DF
reload_df = False
if True:
    interval = 15
    p = cf.p_data / 'feather/df.ftr'

    # read from db or csv
    if reload_df or not p.exists() or dt.fromtimestamp(p.stat().st_mtime) < dt.now() + delta(days=-1):
        log.info('Downloading from db')

        em = ExchangeManager()

        df = Tickers().get_df(
            symbol='XBTUSD',
            startdate=dt(2017, 1, 1),
            interval=interval,
            funding=True,
            funding_exch=em.default('bitmex', refresh=False))

        df.reset_index(drop=False).to_feather(p)
    else:
        df = data.default_df()

    df_ohlc = df.copy()
    log.info(f'Loaded df: [{df.shape[0]:,.0f}] {df.index.min()} - {df.index.max()}')

# %% - ADD SIGNALS
# --%%prun -s cumulative -l 40
# --%%time
use_important = True

if True:
    name = 'lgbm'
    cfg = md.model_cfg(name)
    regression = False
    sm = sg.SignalManager.default().register(mfm)

    n_periods = cf.dynamic_cfg()['target_n_periods']
    # n_periods = 10
    p_ema = None  # 6
    # Target = sg.TargetMeanEMA
    # Target = sg.TargetMean
    if not regression:
        Target = sg.TargetUpsideDownside
    else:
        Target = sg.TargetRatio

    target_signal = Target(
        n_periods=n_periods,
        regression=regression)

    signals = md.DEFAULT_SIGNALS + [target_signal]  # type: List[Any]

    # drop last rows which we cant set proper target
    # don't need to drop last n_periods rows if positive we aren't fitting on them

    df = sm.add_signals(df=df, signals=signals, use_important_dynamic=use_important)

    if not use_important:
        df_all = df.copy()

    if not regression:
        sk.show_prop(df=df)

# %% - FEATURES, MODELMANAGER
# df = df_all.copy()
# use_important = False
split_date = cf.D_SPLIT
if True:
    n_splits = 5
    train_size = 0.9
    # max_train_size = int(df.shape[0] * train_size / n_splits)
    # max_train_size = 20_000
    max_train_size = None
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

    wm = WeightsManager.from_config(df).register(mfm)

    if regression:
        scoring = dict(rmse='neg_root_mean_squared_error')
        scorer = None
    else:
        scorer = ml.StratScorer()

        scoring = dict(
            acc='accuracy',
            wt=make_scorer(sk.weighted_score, weights=wm.weights),
            max=lambda *args: scorer.score(*args, _type='max'),
            final=lambda *args: scorer.score(*args, _type='final')
        )

    cv_args = dict(cv=cv, n_jobs=-1, return_train_score=True, scoring=scoring)
    mm = md.make_model_manager(name=name, df=df, use_important=use_important, wm=wm)

    x_train, y_train, x_test, y_test = mm.make_train_test(df=df, split_date=split_date)

    cv_args |= dict(
        fit_params=wm.fit_params(x=x_train, name=name),
        return_estimator=True)

    mm.init_cv(scoring=scoring, cv_args=cv_args, scorer=scorer)

    LGBM = LGBMClassifier if not regression else LGBMRegressor
    models = dict(
        lgbm=LGBM(
            num_leaves=40, n_estimators=80, max_depth=10, boosting_type='dart', learning_rate=0.1,
            # device='gpu', max_bins=15
        ))

    # num_leaves=30, n_estimators=100, max_depth=30, boosting_type='dart', learning_rate=0.1)

    # models = dict(
    #     lgbm=LinearRegression(normalize=True))

    mm.init_models(models)

    log.info(f'num_feats: {len(mm.ct.transformers[1][2])}')

# %% - CROSS VALIDATION

if False:

    # --%%time
    #  --%%prun -s cumulative -l 40
    # steps = [
    #     (1, ('pca', PCA(n_components=60, random_state=0)))]
    # steps = None
    mm.cross_val(models)

    if not regression:
        res_dfs = [m.cv_data['df_result'] for m in mm.cv_data[name]]
        scorer.show_summary(dfs=res_dfs, scores=mm.scores[name])


# %% - RUN STRAT

# TODO test iter_predict maxhigh/minlow
# TODO track live retrain runs w prediction acc? ... would only predict for ~day?
# save models to predict later?

is_iter = True

target_signal = Target(n_periods=20)
df = sm.add_signals(df=df_all, signals=target_signal, force_overwrite=True)
df = sm.filter_n_feats(df=df, n=34)

# with mlflow.start_run(experiment_id='0'):
model = LGBMClsLog(
    num_leaves=34,
    n_estimators=97,
    max_depth=36,
    boosting_type='dart',
    learning_rate=0.1)

if not is_iter:
    # if False:
    #     df_bbit = db.get_df('bybit', 'BTCUSD', startdate=dt(2021,1,1))
    #     df2 = df.pipe(live.replace_ohlc, df_bbit)

    df_pred = mm \
        .add_predict(
            df=df,
            weighted_fit=True,
            filter_fit_quantile=0.6,
            name=name,
            proba=not regression)
else:
    # retrain every x hours (24 ish) and predict for the test set
    # NOTE limiting max train data to ~3yrs ish could be helpful
    batch_size = 24 * 4 * 8
    filter_fit_quantile = 0.55
    retrain_feats = False

    df_pred = sk.add_predict_iter(
        df=df,
        # name=name,
        wm=wm,
        model=model,
        batch_size=batch_size,
        split_date=cf.D_SPLIT,
        # min_size=mm.df_train.shape[0],
        max_train_size=None,
        # max_train_size=4*24*365*3,
        filter_fit_quantile=filter_fit_quantile)

df_pred = df_pred \
    .pipe(md.add_proba_trade_signal, regression=regression, n_smooth=4)

strat = ml.make_strat(symbol=cf.SYMBOL, order_offset=-0.0006).register(mfm)

bm = bt.BacktestManager(
    startdate=cf.D_SPLIT,
    strat=strat,
    df=df_pred) \
    .run(prnt=True, plot_balance=True) \
    .register(mfm)


# %% - PLOT
if True:
    periods = 30 * 24 * 4
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
        df=df_pred.pipe(pu.clean_cols, cols + ['target']),
        df_balance=df_balance,
        df_trades=df_trades,
        startdate=startdate,
        periods=periods)

# %% - INIT SHAP MANAGER
# x_shap = x_shap.loc['2021-08-01':]
# y_shap = y_shap.loc['2021-08-01':]
spm = sk.ShapManager(df=df_all, model=mm.models['lgbm'], n_sample=10_000)

# %%
res = spm.shap_n_important(n=45, save=True, upload=False, as_list=True)
cols = res['most']
cols

# %% - SHAP PLOT
spm.plot(plot_type='violin')

# %%
spm.force_plot(sample_n=0)

# %% LGBM Tree Digraph
# spm.check_init()
lgb.create_tree_digraph(
    # spm.model,
    mm.models['lgbm'],
    show_info=['internal_count', 'leaf_count', 'data_percentage'],
    orientation='vertical')

# %% - DENSITY PLOTS
# # %% time
data = mm.ct.fit_transform(x_train)

expr = 'sfp'
df_trans = sk.df_transformed(data=data, ct=mm.ct) \
    .assign(target=y_train) \
    .pipe(pu.select_cols, expr=expr, include='target')

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

# %%prun -D stats.stats
# import pstats
# ps = pstats.Stats('stats.stats')
# ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

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

if False:
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
        refit='rmse',
        # refit='final'
    )
