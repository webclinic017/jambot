import copy
from typing import *

import numpy as np
import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import getlog
from jambot import signals as sg
from jambot.signals import SignalManager
from jgutils import fileops as jfl
from jgutils import pandas_utils as pu

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    from jambot.ml.classifiers import LGBMClsLog
    from jambot.sklearn_utils import ModelManager

log = getlog(__name__)

DEFAULT_SIGNALS = [
    'EMA',
    'Momentum',
    'Trend',
    'Candle',
    'Volatility',
    'Volume']


def model_cfg(name: str) -> Dict[str, Any]:
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
    from jambot.ml.classifiers import LGBMClsLog

    return dict(
        lgbm=dict(
            target=['target'],
            # target_kw=dict(n_periods=10, regression=False),
            target_cls=sg.TargetUpsideDownside,
            # drop_cols=['target_max', 'target_min'],
            model_kw=dict(
                num_leaves=40,
                n_estimators=80,
                max_depth=10,
                boosting_type='dart',
                random_state=0),
            model_cls=LGBMClsLog,
            n_smooth_proba=3,
        ),
        # ridge=dict(
        #     target=['target_max', 'target_min'],
        #     target_kw=dict(n_periods=4),
        #     target_cls=sg.TargetMaxMin,
        #     drop_cols=['target'],
        #     model_kw=dict(estimator=Ridge(random_state=0)),
        #     model_cls=MultiOutputRegressor,
        # )
    ).get(name, {})


def add_signals(
        df: pd.DataFrame,
        name: str,
        symbol: str = SYMBOL,
        drop_ohlc: bool = False,
        use_important_dynamic: bool = True,
        **kw) -> pd.DataFrame:
    """Add signal cols + target to df
    - group by symbol

    Parameters
    ----------
    df : pd.DataFrame
        raw df with no features
    name : str
        model name
    symbol : str
        if MULTI_ALTS will add default signals with same params for all alts
    drop_ohlc : bool
        drop ohlcv columns from raw df, default False
    use_important_dynamic : bool
        use up to n most important feats, from dynamic config file
    use_important : bool
        exclude least important columns from model, default True

    Returns
    -------
    pd.DataFrame
        df with features added
    """
    target_signal = sg.TargetUpsideDownside.from_config(symbol=symbol)

    signals = DEFAULT_SIGNALS + [target_signal]

    return df.groupby('symbol', group_keys=False) \
        .apply(lambda df: SignalManager.default().add_signals(
            df=df,
            signals=signals,
            use_important_dynamic=use_important_dynamic,
            drop_ohlc=drop_ohlc,
            symbol=symbol,
            **kw))


def make_model_manager(
        name: str,
        df: pd.DataFrame,
        use_important: bool = False,
        **kw) -> 'ModelManager':
    """Instantiate ModelManager
    - NOTE not used

    Parameters
    ----------
    name : str
        model name
    df : pd.DataFrame
        df with signals added
    use_important : bool
        filter to important cols only

    Returns
    -------
    ModelManager
    """
    cfg = model_cfg(name)
    target = cfg.get('target')

    drop_cols = copy.copy(cf.DROP_COLS)

    # only use n most imporant features from shap_vals
    if use_important:
        drop_cols += jfl.load_pickle(p=cf.p_data / 'feats/least_imp_cols.pkl')

    # for azure training, drop cols will have been dropped first to save memory
    drop_cols = [c for c in drop_cols if c in df.columns]

    features = dict(
        target=target,
        drop=drop_cols)

    features['numeric'] = pu.all_except(df, features.values())

    encoders = dict(
        drop='drop',
        # numeric=MinMaxScaler(feature_range=(0, 1))
        numeric='passthrough'
    )

    from jambot.sklearn_utils import ModelManager

    return ModelManager(
        features=features,
        encoders=encoders, **kw)


def make_model(name: str, symbol: str = SYMBOL) -> 'LGBMClsLog':
    """Create instance of LGBMClassifier

    Parameters
    ----------
    name : str

    symbol : str, optional
        default SYMBOL

    Returns
    -------
    LGBMClassifier
    """
    cfg = model_cfg(name)
    model = cfg['model_cls']  # type: LGBMClsLog
    return model.from_config(symbol=symbol)


def make_pipeline(name: str, df: pd.DataFrame) -> 'Pipeline':
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
    model = make_model(name=name)

    return mm.make_pipe(name=name, model=model, steps=None)


def df_proba(x: pd.DataFrame, model: 'LGBMClsLog', **kw) -> pd.DataFrame:
    """Return df of predict_proba, with timestamp index

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : 'BaseEstimator'
        fitted model/pipeline

    Returns
    -------
    pd.DataFrame
        df with proba_ added
    """
    arr = model.predict_proba(x)
    m = {-1: 'short', 0: 'neutral', 1: 'long'}
    cols = [f'proba_{m.get(c)}' for c in model.classes_]
    return pd.DataFrame(data=arr, columns=cols, index=x.index)


def df_y_pred(x: pd.DataFrame, model: 'BaseEstimator') -> pd.DataFrame:
    """Return df with y_pred added

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : 'BaseEstimator'
        model/pipe

    Returns
    -------
    pd.DataFrame
        df with y_pred added
    """
    return pd.DataFrame(
        data=model.predict(x),
        columns=['y_pred'],
        index=x.index)


def add_probas(df: pd.DataFrame, model: 'BaseEstimator', x: pd.DataFrame, do: bool = True, **kw) -> pd.DataFrame:
    """Convenience func to add probas if don't exist

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : 'BaseEstimator'
        model/pipe
    x : pd.DataFrame
        section of df to predict on
    do : bool, optional
        pass False to skip (for regression)

    Returns
    -------
    pd.DataFrame
        df with proba_ added
    """

    # already added probas
    if not do or 'proba_long' in df.columns:
        return df

    return df.pipe(pu.left_merge, df_right=df_proba(x=x, model=model))


def add_y_pred(df: pd.DataFrame, model: 'BaseEstimator', x: pd.DataFrame) -> pd.DataFrame:
    """Add y_pred col to df with model.predict

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : 'BaseEstimator'
        model/pipe
    x : pd.DataFrame
        section of df to predict on

    Returns
    -------
    pd.DataFrame
        df with y_pred added
    """
    return df.pipe(pu.left_merge, df_right=df_y_pred(x=x, model=model))


def convert_proba_signal(
        df: pd.DataFrame,
        col: str = 'rolling_proba',
        regression: bool = False) -> pd.DataFrame:
    """Convert probas btwn 0-1 to a signal of 0, 1 or -1 with threshold 0.5

    Parameters
    ----------
    df : pd.DataFrame
        df with rolling proba col
    col : str
        col to conver to signal, default 'rolling_proba'

    Returns
    -------
    pd.DataFrame
        df with signal added
    """
    s = df[col]

    # for regression, just using y_pred which is already pos/neg, not proba btwn 0-1
    offset = 0.5 if not regression else 0

    return df \
        .assign(
            signal=np.sign(np.diff(np.sign(s - offset), prepend=np.array([0]))))


def add_proba_trade_signal(
        df: pd.DataFrame,
        regression: bool = False,
        n_smooth: int = None, **kw) -> pd.DataFrame:
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

    # NOTE not implemented yet, need to make func dynamic for regression
    rolling_col = 'proba_long' if not regression else 'y_pred'
    n_smooth = n_smooth or cf.dynamic_cfg()['n_periods_smooth']  # get dynamic config value

    return df \
        .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba') \
        .pipe(convert_proba_signal, regression=regression)


def add_preds_probas(df: pd.DataFrame, pipe: 'BaseEstimator', **kw) -> pd.DataFrame:
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
    x = pu.safe_drop(df, 'target')

    # NOTE y_pred only needed for scoring to get accuracy, not for strategy
    return df \
        .pipe(add_y_pred, model=pipe, x=x) \
        .pipe(add_probas, model=pipe, x=x) \
        .pipe(add_proba_trade_signal, **kw)
