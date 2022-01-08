from typing import *

import mlflow
import pandas as pd
import skopt
import yaml
from skopt import plots
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Real

from jambot import config as cf
from jambot import data, getlog
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

ACTIVE_RESULT = 8
p_skopt = cf.p_data / 'skopt'
p_res = p_skopt / f'results{ACTIVE_RESULT}.pkl'


def get_space() -> List['Dimension']:
    space = [
        Integer(6, 40, name='max_depth'),
        Integer(20, 100, name='num_leaves'),
        Integer(40, 100, name='n_estimators'),
        Integer(1, 20, name='n_smooth'),
        Integer(2, 40, name='n_target'),
        Integer(20, 50, name='num_feats'),
        Real(0.2, 0.9, name='filter_fit_quantile', prior='uniform')
    ]  # type: List[Dimension]

    return space


def objective(
        df: pd.DataFrame,
        mfm: MlflowManager,
        wm: WeightsManager,
        df_pred: pd.DataFrame = None,
        sm: SignalManager = None,
        **kw) -> float:

    log.info(f'objective_kw: {kw}')
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
        split_date=cf.D_SPLIT)  # type: Dict[str, Any]

    if 'n_target' in kw and not sm is None:
        # TODO cache this series somehow
        target_signal = Target(n_periods=kw['n_target'])
        df = sm.add_signals(df=df, signals=target_signal, force_overwrite=True)

    if 'num_feats' in kw and not sm is None:
        df = sm.filter_n_feats(df=df, n=kw['num_feats'])

    if df_pred is None:
        df_pred = sk.add_predict_iter(
            df=df, wm=wm, model=model, **iter_kw)

    with mlflow.start_run(experiment_id='3'):
        df_pred = df_pred \
            .pipe(md.add_proba_trade_signal, n_smooth=n_smooth)

        strat = make_strat(symbol=cf.SYMBOL, order_offset=-0.0006).register(mfm)

        bm = BacktestManager(
            startdate=cf.D_SPLIT,
            strat=strat,
            df=df_pred) \
            .run(prnt=True, plot_balance=False) \
            .register(mfm)

        scores = dict(
            acc=sk.accuracy_score(df_pred.target, df_pred.y_pred),
            w_acc=sk.weighted_score(df_pred.target, df_pred.y_pred, wm.weights))

        metrics = scores | dict(
            batch_size=iter_kw['batch_size'],
            filter_fit_quantile=iter_kw['filter_fit_quantile'],
            n_periods_smooth=n_smooth,
            interval=cf.INTERVAL)

        params = dict(
            is_iter=True,
            retrain_feats=iter_kw['retrain_feats'])

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mfm.log_all(flush=True)

    return -1 * bm.df_result['ci_monthly'].iloc[0]


def run_opt(n_calls: int = 10, df: pd.DataFrame = None, mfm: MlflowManager = None) -> 'OptimizeResult':

    p_check = p_skopt / 'checkpoint.pkl'

    if mfm is None:
        mfm = MlflowManager()

    # SignalManager
    sm = SignalManager.default().register(mfm)

    if df is None:
        target_signal = Target(n_periods=10)
        signals = md.DEFAULT_SIGNALS + [target_signal]  # type: ignore
        df = sm.add_signals(
            df=data.default_df(), signals=signals, use_important=True)

    # WeightsManager
    wm = WeightsManager.from_config(df).register(mfm)

    space = get_space()

    @skopt.utils.use_named_args(space)
    def _objective(**kw):
        return objective(df=df, mfm=mfm, wm=wm, sm=sm, **kw)

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
        # acq_func='LCB',
        acq_func='gp_hedge',
        # acq_func='PI',
        # xi=0.01,
        kappa=3,
        verbose=True,
        n_calls=n_calls,
        # random_state=0,
        n_initial_points=30,
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


def get_best_results(df: pd.DataFrame = None, top_n: int = 50, exp_id: str = '3') -> dict[str, float]:
    """Get dict of median of best params to save to static config file

    Parameters
    ----------
    df : pd.DataFrame, optional
        df of mlflow results, default None
    top_n : int, optional
        top n results to use, default 50

    Returns
    -------
    dict[str, float]
    >>> {'max_depth': 34.5,
        'n_estimators': 91.0,
        'num_leaves': 60.0,
        'num_feats': 32.0,
        'n_periods_smooth': 3.0,
        'target_n_periods': 20.0,
        'filter_fit_quantile': 0.5154932627849301}
    """
    if df is None:
        mfm = MlflowManager()
        df = mfm.df_results(experiment_ids=exp_id)

    # TODO put this in config somewhere? idk
    cols = ['max_depth', 'n_estimators', 'num_leaves', 'num_feats',
            'n_periods_smooth', 'target_n_periods', 'filter_fit_quantile']

    int_cols = df.select_dtypes(int).columns.tolist()

    return df \
        .reset_index() \
        .drop_duplicates('ci_monthly') \
        .sort_values('ci_monthly', ascending=False) \
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
