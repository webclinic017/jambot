import time
from collections import defaultdict
from typing import *

import mlflow
import pandas as pd
import skopt
import yaml
from skopt import plots
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Real

from jambot import config as cf
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.ml import models as md
from jambot.ml.classifiers import LGBMClsLog
from jambot.signals import SignalManager
from jambot.signals import TargetUpsideDownside as Target
from jambot.tradesys.backtest import BacktestManager
from jambot.tradesys.strategies.ml import make_strat
from jambot.utils.mlflow import MlflowManager
from jambot.weights import WeightsManager
from jgutils import pandas_utils as pu
from jgutils.functions import nested_dict_update

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult
    from skopt.space import Dimension

log = getlog(__name__)

ACTIVE_RESULT = 14
MLFLOW_EXP = str(9)  # TODO make this set auto with new run
p_skopt = cf.p_data / 'skopt'
p_res = p_skopt / f'results{ACTIVE_RESULT}.pkl'


class ObjectCache(object):
    """Cachce unique objects between optimization calls to avoid recalc"""

    def __init__(self):
        self.obj_cache = defaultdict(dict)

    @staticmethod
    def make_key(params: Dict[str, Any]) -> str:
        """Create key from params dict"""
        return ','.join(f'{k}={v}' for k, v in params.items())

    def get_or_save(self, obj_cls: Type[Any], params: Dict[str, Any], **kw) -> Any:
        """Get objc if exists in cache, or create new

        Parameters
        ----------
        obj_cls : Type[Any]
            uninstantiated object class
        params : Dict[str, Any]
            unique params to save obj with
        kw : dict
            extra non param_key args to instantiate object with

        Returns
        -------
        Any
            instantiated object
        """
        name = obj_cls.__name__
        params_key = self.make_key(params)

        # check if obj exists in cache
        obj = self.obj_cache.get(name, {}).get(params_key, None)

        # create obj and save
        if obj is None:
            obj = obj_cls(**params, **kw)
            self.obj_cache[name] = {params_key: obj}
        #     log.info(f'Init new object: {name}, {params}')
        # else:
        #     log.info(f'Using cached object: {name}, {params}')

        return obj


def get_space() -> List['Dimension']:
    space = [
        Integer(6, 60, name='max_depth'),
        Integer(20, 100, name='num_leaves'),
        Integer(40, 100, name='n_estimators'),
        Integer(1, 10, name='n_smooth'),
        Integer(2, 40, name='n_target'),
        Integer(2, 40, name='weights_n_periods'),
        Integer(20, 80, name='num_feats'),
        # Integer(12, 12 * 4, name='max_train_size'),
        Real(0.2, 0.9, name='filter_fit_quantile', prior='uniform'),
        Real(-0.002, -0.0001, name='order_offset', prior='uniform')
    ]  # type: List[Dimension]

    return space


def objective(
        df: pd.DataFrame,
        mfm: MlflowManager,
        obc: ObjectCache,
        wm: WeightsManager = None,
        df_pred: pd.DataFrame = None,
        sm: SignalManager = None,
        n_jobs: int = -1,
        **kw) -> Union[float, Tuple[float, float]]:

    # time penalty
    start = time.time()

    str_kw = ', '.join([f'{k}={v}' for k, v in kw.items()])  # cleaner log message
    log.info(str_kw)
    cfg = md.model_cfg('lgbm')

    n_smooth = kw.get('n_smooth', cfg['n_smooth_proba'])

    model = LGBMClsLog(
        num_leaves=kw.get('num_leaves', 40),
        n_estimators=kw.get('n_estimators', 80),
        max_depth=kw.get('max_depth', 10),
        boosting_type='dart',
        learning_rate=0.1).register(mfm)

    iter_kw = dict(
        batch_size=24 * 4 * 8,
        filter_fit_quantile=kw.get('filter_fit_quantile', 0.6),
        retrain_feats=False,
        split_date=cf.D_SPLIT,
        max_train_size=kw.get('max_train_size', 48))  # type: Dict[str, Any]

    # create target signal
    if 'n_target' in kw and not sm is None:
        params = dict(n_periods=kw['n_target'])
        target_signal = obc.get_or_save(Target, params=params, cache=True)
        df = sm.add_signals(df=df, signals=target_signal, force_overwrite=True)

    # filter df_all signals to select features
    if 'num_feats' in kw and not sm is None:
        df = sm.filter_n_feats(df=df, n=kw['num_feats'])

    # create WeightsManager
    if wm is None and 'weights_n_periods' in kw:
        params = dict(n_periods=kw['weights_n_periods'], weight_linear=True)
        wm = obc.get_or_save(WeightsManager, params=params, df=df).register(mfm)

    # add model predictions iteratively
    if df_pred is None:
        df_pred = sk.add_predict_iter(
            df=df.iloc[:-target_signal.n_periods],
            wm=wm,
            model=model,
            n_jobs=n_jobs,
            **iter_kw)

    with mlflow.start_run(experiment_id=MLFLOW_EXP):
        df_pred = df_pred \
            .pipe(md.add_proba_trade_signal, n_smooth=n_smooth)

        strat = make_strat(
            symbol=cf.SYMBOL,
            order_offset=kw.get('order_offset', -0.0006)).register(mfm)

        try:
            bm = BacktestManager(
                startdate=cf.D_SPLIT,
                strat=strat,
                df=df_pred) \
                .run(prnt=True, plot_balance=False) \
                .register(mfm)

            scores = dict(
                acc=sk.accuracy_score(df_pred.target, df_pred.y_pred),
                w_acc=sk.weighted_score(df_pred.target, df_pred.y_pred, wm.weights))

            iter_keys = ['batch_size', 'filter_fit_quantile', 'max_train_size']
            metrics = scores | dict(
                n_periods_smooth=n_smooth,
                interval=cf.INTERVAL) \
                | {k: iter_kw[k] for k in iter_keys}

            params = dict(
                is_iter=True,
                retrain_feats=iter_kw['retrain_feats'])

            mlflow.log_metrics(metrics)
            mlflow.log_params(params)
        except Exception as e:
            log.error('filed backtest')
            print(e)

        mfm.log_all(flush=True)

    # return -1 * bm.df_result['ci_monthly'].iloc[0]
    # return -1 * strat.wallet.sharpe(weighted=True) #, time.time() - start
    return -1 * strat.wallet.ci_monthly(weighted=True)


def run_opt(
        df: pd.DataFrame,
        n_calls: int = 10,
        mfm: MlflowManager = None,
        n_jobs: int = -1) -> 'OptimizeResult':

    p_check = p_skopt / 'checkpoint.pkl'
    obc = ObjectCache()  # save Target and WeightsManager to avoid recalc

    if mfm is None:
        mfm = MlflowManager()

    # SignalManager
    sm = SignalManager.default().register(mfm)

    # NOTE dont use this, use df_all, cols will be filtered by num_feats
    # if df is None:
    #     target_signal = Target(n_periods=10)
    #     signals = md.DEFAULT_SIGNALS + [target_signal]  # type: ignore
    #     df = sm.add_signals(
    #         df=data.default_df(), signals=signals, use_important=True)

    space = get_space()

    @skopt.utils.use_named_args(space)
    def _objective(**kw):
        return objective(df=df, mfm=mfm, obc=obc, wm=None, sm=sm, n_jobs=n_jobs, **kw)

    checkpoint = CheckpointSaver(p_check, store_objective=False)

    if p_res.exists():
        res_old = skopt.load(p_res)
        x0 = res_old.x_iters
        y0 = res_old.func_vals
    else:
        x0, y0 = None, None

    results = skopt.gp_minimize(
        func=_objective,
        dimensions=space,
        # base_estimator='ET',
        callback=[checkpoint],
        # acq_func='EIps',
        acq_func='gp_hedge',
        # acq_func='PI',
        # xi=0.05,
        kappa=3.5,
        verbose=True,
        n_calls=n_calls,
        n_initial_points=min(n_calls, 30),
        x0=x0,
        y0=y0)

    # dump results
    skopt.dump(results, p_res, store_objective=False)

    return results


def load_res() -> 'OptimizeResult':
    return skopt.load(p_res)


def plot_eval():
    res = load_res()
    plots.plot_evaluations(res)


def plot_conv():
    res = load_res()
    plots.plot_convergence(res)


def get_best_results(
        df: pd.DataFrame = None,
        top_n: int = 50,
        exp_id: str = MLFLOW_EXP,
        by: str = 'ci_monthly') -> Dict[str, dict[str, float]]:
    """Get dict of median of best params to save to static config file
    - TODO find better way to avg best results. weighted mean?

    Parameters
    ----------
    df : pd.DataFrame, optional
        df of mlflow results, default None
    top_n : int, optional
        top n results to use, default 50

    Returns
    -------
    dict[str, float]
    >>> {'XBTUSD':
            {'max_depth': 34,
            'n_estimators': 91,
            'num_leaves': 60,
            'num_feats': 32,
            'n_periods_smooth': 3,
            'target_n_periods': 20,
            'filter_fit_quantile': 0.5154932627849301}}
    """
    if df is None:
        mfm = MlflowManager()
        df = mfm.df_results(experiment_ids=exp_id)

    # TODO put this in config somewhere? idk
    cols = [
        'max_depth', 'n_estimators', 'num_leaves',
        'num_feats', 'n_periods_smooth',
        'target_n_periods', 'filter_fit_quantile',
        'weights_n_periods', 'order_offset']

    int_cols = df.select_dtypes(int).columns.tolist()

    return df \
        .reset_index() \
        .drop_duplicates('ci_monthly') \
        .sort_values(by=by, ascending=False) \
        .head(top_n) \
        .groupby('symbol')[cols] \
        .median() \
        .pipe(pu.convert_dtypes, cols=int_cols, _type=int) \
        .to_dict(orient='index')


def write_best_results(m_res: Dict[str, float] = None, **kw) -> None:
    """Write best opt results to model_config.yaml
    - TODO also upload this to azure?

    Parameters
    ----------
    m_res : Dict[str, float], optional
        new results to merge to existing model_config, default None
    """
    p = cf.p_res / 'model_config.yaml'

    # get results if none passed in
    if m_res is None:
        m_res = get_best_results(**kw)

    # get existing vals
    with open(p, 'r') as file:
        m_in = yaml.full_load(file)

    # update existing with new
    m_out = nested_dict_update(m_in, m_res)

    # write updated dict back to file
    with open(p, 'w') as file:
        yaml.dump(m_out, file)
