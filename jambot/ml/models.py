import copy

import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA  # noqa
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot import signals as sg
from jambot import sklearn_utils as sk

# from sklearn.preprocessing import MinMaxScaler


log = getlog(__name__)


def model_cfg(name: str) -> dict:
    """Config for models

    Parameters
    ----------
    name : str
        model name

    Returns
    -------
    dict
        specified model config options
    """

    return dict(
        lgbm=dict(
            target=['target'],
            target_kw=dict(n_periods=10, regression=False),
            target_cls=sg.TargetUpsideDownside,
            # drop_cols=['target_max', 'target_min'],
            model_kw=dict(
                num_leaves=100,
                n_estimators=100,
                max_depth=20,
                boosting_type='dart',
                random_state=0),
            model_cls=LGBMClassifier,
            n_smooth_proba=3,
            n_periods_weighted=8
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


def add_signals(
        df: pd.DataFrame,
        name: str,
        drop_ohlc: bool = False,
        use_important: bool = True) -> pd.DataFrame:
    """Add signal cols to df

    Parameters
    ----------
    df : pd.DataFrame
        raw df with no features
    name : str
        model name
    drop_ohlc : bool
        drop ohlcv columns from raw df, default False
    use_important : bool
        exclude least important columns from model, default True

    Returns
    -------
    pd.DataFrame
        df with features added
    """
    cfg = model_cfg(name)

    target_signal = cfg['target_cls'](**cfg['target_kw'])

    signals = [
        'EMA',
        'Momentum',
        'Trend',
        'Candle',
        'Volatility',
        'Volume',
        target_signal]

    return sg.SignalManager(**cf.config['signalmanager_kw']) \
        .add_signals(df=df, signals=signals, use_important=use_important) \
        .pipe(f.safe_drop, cols=cf.config['drop_cols'], do=drop_ohlc)


def make_model_manager(name: str, df: pd.DataFrame, use_important: bool = False) -> 'sk.ModelManager':
    """Instantiate ModelManager

    Parameters
    ----------
    name : str
        model name
    df : pd.DataFrame
        df with signals added

    Returns
    -------
    sk.ModelManager
    """
    cfg = model_cfg(name)
    target = cfg.get('target')

    drop_cols = copy.copy(cf.config['drop_cols'])

    # only use n most imporant features from shap_vals
    if use_important:
        drop_cols += f.load_pickle(p=cf.p_data / 'feats/least_imp_cols.pkl')

    # for azure training, drop cols will have been dropped first to save memory
    drop_cols = [c for c in drop_cols if c in df.columns]

    features = dict(
        target=target,
        drop=drop_cols)

    features['numeric'] = sk.all_except(df, features.values())

    encoders = dict(
        drop='drop',
        # numeric=MinMaxScaler(feature_range=(0, 1))
        numeric='passthrough'
    )

    return sk.ModelManager(
        features=features,
        encoders=encoders)


def make_model(name: str) -> BaseEstimator:
    cfg = model_cfg(name)

    # init model with kws
    cls = cfg['model_cls']
    model = cls(**cfg['model_kw'])
    return model


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
    mm = make_model_manager(name=name, df=df)

    cfg = model_cfg(name)

    # init model with kws
    cls = cfg['model_cls']
    model = cls(**cfg['model_kw'])

    # steps = [
    #     (1, ('pca', PCA(n_components=30, random_state=0)))]
    steps = None

    return mm.make_pipe(name=name, model=model, steps=steps)


def add_preds_probas(df: pd.DataFrame, pipe: BaseEstimator, **kw) -> pd.DataFrame:
    """Convenience func to add y_pred, predict_probas, and trade signal cols to df
    - specific to LGBM model with smoothed rolling probas

    Parameters
    ----------
    df : pd.DataFrame
        df with features, will drop target if exists
    pipe : BaseEstimator
        estimator with predict and predict_proba
    regression : bool
        defines which col to smooth

    Returns
    -------
    pd.DataFrame
        "df_pred" with y_pred and proba_ for each predicted class
    """
    x = f.safe_drop(df, 'target')

    # NOTE y_pred only needed for scoring to get accuracy, not for strategy
    return df \
        .pipe(sk.add_y_pred, model=pipe, x=x) \
        .pipe(sk.add_probas, model=pipe, x=x) \
        .pipe(add_proba_trade_signal, **kw)


def add_proba_trade_signal(df: pd.DataFrame, regression: bool = False, n_smooth: int = None, **kw) -> pd.DataFrame:
    """Convert probas into trade signal for strategy

    Parameters
    ----------
    df : pd.DataFrame
        df with probas added

    Returns
    -------
    pd.DataFrame
        df with trade signal
    """
    cfg = model_cfg('lgbm')

    # NOTE not implemented yet, need to make func dynamic for regression
    rolling_col = 'proba_long' if not regression else 'y_pred'
    n_smooth = n_smooth or cfg['n_smooth_proba']

    return df \
        .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba') \
        .pipe(sk.convert_proba_signal, regression=regression)
