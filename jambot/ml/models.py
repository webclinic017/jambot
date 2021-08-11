from datetime import datetime as dt

import pandas as pd
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from jambot import functions as f
from jambot import getlog
from jambot import signals as sg
from jambot import sklearn_utils as sk
from jambot.database import db


def get_model_params(name: str) -> dict:

    return dict(
        lgbm=dict(
            target=['target'],
            target_kw=dict(n_periods=10, regression=False),
            target_cls=sg.TargetMean,
            drop_cols=['target_max', 'target_min'],
            model_kw=dict(
                num_leaves=50,
                n_estimators=50,
                max_depth=10,
                boosting_type='dart',
                random_state=0),
            model_cls=LGBMClassifier,
        ),
        ridge=dict(
            target=['target_max', 'target_min'],
            target_kw=dict(n_periods=4),
            target_cls=sg.TargetMaxMin,
            drop_cols=['target'],
            model_kw=dict(estimator=Ridge(random_state=0)),
            model_cls=MultiOutputRegressor,
        )
    ).get(name)


def add_signals(df, name: str) -> pd.DataFrame:
    """Add signal cols to df"""
    cfg = get_model_params(name)

    target_signal = cfg['target_cls'](**cfg['target_kw'])

    signals = [
        'EMA',
        'Momentum',
        'Trend',
        'Candle',
        # 'EMASlope',
        # 'Volatility',
        # 'Volume',
        target_signal
    ]

    sm = sg.SignalManager(add_slope=5)

    return sm.add_signals(df=df, signals=signals)
    # .iloc[:-1 * n_periods, :] \
    # .fillna(0)


def make_pipeline(name: str, df: pd.DataFrame) -> Pipeline:
    """Create pipeline to fit/predict on data

    Parameters
    ----------
    name : str
        model name
    df : pd.DataFrame
        df with signals added

    Returns
    -------
    Pipeline
        pipeline with ColumnTransformer, PCA, Model
    """
    cfg = get_model_params(name)

    cols_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    ema_cols = ['ema50', 'ema200']
    drop_feats = cols_ohlcv + ema_cols + cfg.get('drop_cols')  # drop other target cols
    target = cfg.get('target')

    features = dict(
        target=target,
        drop=drop_feats)

    features['numeric'] = sk.all_except(df, features.values())

    encoders = dict(
        drop='drop',
        numeric=MinMaxScaler(feature_range=(0, 1)))

    # init model with kws
    cls = cfg['model_cls']
    model = cls(**cfg['model_kw'])

    steps = [
        (1, ('pca', PCA(n_components=30, random_state=0)))]

    mm = sk.ModelManager(
        features=features,
        encoders=encoders)

    pipe = mm.make_pipe(name=name, model=model, steps=steps)

    return pipe
