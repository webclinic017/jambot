#%%
# Imports
if True:
    import json
    import os
    import sys
    from datetime import datetime as dt
    from datetime import timedelta as delta
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
    from IPython.display import display
    from lightgbm.sklearn import LGBMClassifier
    # import sklearn_helper_funcs as sf
    # from lightgbm.sklearn import LGBMClassifier
    # data
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
                                make_scorer, recall_score)
    from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                        cross_val_score, cross_validate,
                                        train_test_split)
    from sklearn.pipeline import Pipeline, make_pipeline
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

#%%
# load df
kw = dict(
    symbol='XBTUSD',
    daterange=365 * 3,
    startdate=dt(2018, 1, 1),
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

print(f'Start Date: {df.index.min()}')

# %%
# Add features to df
n_periods = 6
target_signal = sg.TargetClass(p_ema=10, n_periods=n_periods, pct_min=0.000000005)

signals = [
    'EMA',
    'EMASlope',
    'Volatility',
    'MACD',
    'RSI',
    sg.Trend(speed=(24, 18)),
    'Candle',
    'SFP',
    target_signal
    ]

df = df.pipe(sg.add_signals, signals) \
    .dropna() \
    .iloc[:-1 * n_periods, :] # drop last rows which we cant set proper target

sf.show_prop(df=df)

#%%

drop_feats = ['Open', 'High', 'Low', 'Close', target_signal.ema_col, 'ema50', 'ema200', 'pxhigh', 'pxlow']

features = dict(
    target=['target'],
    drop=drop_feats,
    passthrough=['ema_trend', 'ema_slope', 'cdl_side', 'macd_trend'])

features['passthrough'] = features['passthrough'] + [col for col in df.columns if any(item in col for item in ('sfp', 'prevhigh', 'prevlow'))]
features['numeric'] = [col for col in df.columns if not any(col in lst for lst in features.values())]

sf.pretty_dict(features)

#%%
encoders = dict(
    drop='drop',
    passthrough='passthrough',
    # binary=OneHotEncoder(),
    # categorical=OneHotEncoder(),
    numeric=MinMaxScaler())

ct = ColumnTransformer(
    transformers=[(name, model, features[name]) for name, model in encoders.items()])
# ct

scoring = dict(acc='accuracy') #, f1=f1_score, average='macro')
cv_args = dict(cv=5, n_jobs=-2, return_train_score=True, scoring=scoring)
mm = sf.ModelManager(ct=ct, scoring=scoring, cv_args=cv_args)

#%% - TRAIN TEST SPLIT
X_train, y_train, X_test, y_test = mm.make_train_test(df=df, target=features['target'], train_size=0.9, shuffle=False)

#%% - COLUMN TRANSFORMED
run = False
if run:
    data = ct.fit_transform(X_train)
    df_trans = sf.df_transformed(data=data, ct=ct)
    print(df_trans.shape)
    display(df_trans.describe().T)

#%%
models = dict(
    dummy=DummyClassifier(strategy='most_frequent'))
    
mm.cross_val(models)

#%% - MODELS
models = dict(
    # rnd_forest=RandomForestClassifier,
    # xgb=XGBClassifier,
    # ada=AdaBoostClassifier(),
    lgbm=LGBMClassifier() #num_leaves=30, n_estimators=100, max_depth=30)
    )
    
mm.cross_val(models)

#%% - GRID - LGBMClassifier
params = dict(
    boosting_type=['gbdt', 'dart', 'goss', 'rf'],
    n_estimators=[25, 50, 100, 200],
    max_depth=[-1, 5, 10, 20, 30],
    num_leaves=[5, 10, 20, 40, 100],
    )

grid = mm.search(
    name='lgbm',
    params=params,
    search_type='random',
    refit='acc')

#%% - SAVE MODEL
mm.save_model('lgbm', best_est=True)

#%% - ADD PREDICTIONS
name = 'lgbm'
# name = 'ada'
# model = mm.load_model(name)
# df_pred = mm.add_predict(df=df, model=model)
df_pred = mm.add_predict(df=df, name=name, best_est=True)

#%% - CLASSIFICATION REPORT
mm.class_rep('lgbm', best_est=True)

#%% GRID - AdaBoostClassifier
params = dict(
    algorithm=['SAMME', 'SAMME.R'],
    n_estimators=[50, 100, 200, 400],
    learning_rate=[0.0125, 0.25, 0.5, 1, 2])

grid = mm.search(
    name='ada',
    params=params,
    search_type='random',
    refit='acc')

#%% - SHAP PLOT
# model = mm.get_model(name='lgbm', best_est=True)
model = mm.models['lgbm']
# model.fit_transform
explainer, shap_values, X_sample, X_enc = sf.shap_explainer_values(X=X_train, y=y_train, ct=ct, model=model)

# show shap plot
shap.summary_plot(
    shap_values=shap_values[0],
    features=X_sample, # X_train_sample
    plot_type='violin',
    axis_color='white')

#%% - SHAP FORCE PLOT
shap.initjs()

explainer, shap_values, X_sample, X_enc = sf.shap_explainer_values(X=X_test, y=y_test, ct=ct, model=model)

n = 0
shap.force_plot(
    explainer.expected_value[0], shap_values[0][n, :], X_enc.iloc[n, :])

#%% - RUN STRAT
strat = ml.Strategy(lev=5, min_proba=0.5, min_agree=6, slippage=0, stoppercent=-0.025)
idx = mm.df_test.index

kw = dict(
    symbol='XBTUSD',
    # daterange=365 * 3,
    startdate=idx[0],
    # interval=1
    )

print(f'Test range: {idx[0]} - {idx[-1]}')
sym = bt.Backtest(**kw, strats=strat, df=df_pred)
sym.decide_full()
# f.print_time(start)
sym.print_final()
sym.account.plot_balance(logy=True)
trades = strat.trades
t = trades[-1]

#%% - PLOT

periods = 600
# periods = 200
startdate = dt(2020, 9, 20)

fig = ch.chart(
    df=df_pred[df_pred.index >= X_test.index[0]],
    periods=periods,
    last=True,
    startdate=startdate,
    df_balance=strat.a.df_balance)
fig.show()

#%% - RIDGE
models = dict(
    lgbm_rfecv=mm.models['lgbm'])

extra_steps = (1, ('rfecv', RFECV(Ridge()))) # must insert rfecv BEFORE other model
mm.cross_val(models, steps=extra_steps)

#%%

pipe_rfe = mm.pipes['lgbm_rfecv']
pipe_rfe.fit(X_train, y_train) # need fit pipe again outside of cross_validate
rfecv = pipe_rfe.named_steps['rfecv']
rfecv.n_features_

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
pipe_rfe.fit(X_train, y_train)
rfecv = pipe_rfe.named_steps['rfecv']
rfecv.n_features_