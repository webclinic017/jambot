import copy
import re
import sqlite3
import struct
import time
from abc import ABCMeta, abstractproperty
from collections import Counter, deque
from pathlib import Path
from typing import *

import pandas as pd
import sqlalchemy as sa
import yaml
from aenum import StrEnum
from pandas.api.types import is_numeric_dtype
from sqlalchemy.orm import Session, sessionmaker

from jambot import config as cf
from jambot import getlog
from jambot.common import DictRepr
from jgutils import functions as jf
from jgutils import pandas_utils as pu

# don't need mlflow package for azure app
if not cf.AZURE_WEB:
    import mlflow
    from mlflow.entities.run import Run
    from mlflow.entities.run_status import RunStatus
    from mlflow.entities.source_type import SourceType
    from mlflow.store.tracking.dbmodels.models import (SqlLatestMetric,
                                                       SqlParam, SqlRun,
                                                       SqlTag)
    from mlflow.tracking import MlflowClient
    from mlflow.tracking.fluent import _get_or_start_run
else:
    SqlLatestMetric = ''
    SqlParam = ''
    SqlRun = ''
    SqlTag = ''

log = getlog(__name__)


class MlflowLoggable(object, metaclass=ABCMeta):
    """Class to ensure object provides params/metrics for mlflow logging"""

    def register(self, mfm: 'MlflowManager'):
        """Attach self to mf manager

        Parameters
        ----------
        mfm : MlflowManager

        Returns
        -------
            self
        """
        mfm.register(self)
        return self

    @abstractproperty
    def log_items(self) -> Dict[str, Any]:
        """Log items (str or num) for mlflow"""
        pass

    def get_metrics_params(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Convert self.log_items to params, metrics for mlflow logging

        Returns
        -------
        Tuple[Dict[str, str], Dict[str, Any]]
            dicts of params, metrics
        """
        metrics, params = {}, {}

        for k, v in self.log_items.items():
            if is_numeric_dtype(type(v)) and not isinstance(v, bool):
                metrics[k] = v
            else:
                params[k] = str(v)

        return params, metrics

    def log_mlflow(self) -> None:
        """Log params, metrics, artifacts, dfs for self and strategy to mlflow"""
        params, metrics = self.get_metrics_params()

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        # NOTE probably need to make this handle multi artifacts
        # if hasattr(self, 'log_artifact'):
        #     mlflow.log_artifact(self.log_artifact())

        # if hasattr(self, 'log_dfs'):
        #     for kw in jf.as_list(self.log_dfs()):
        #         MlflowManager.log_df(**kw)


class MlflowManager(DictRepr):

    indexes = dict(
        df_pred='timestamp',
        df_trades='timestamp')

    class ItemType(StrEnum):
        METRIC: str = 'metric'
        PARAM: str = 'param'
        TAG: str = 'tag'
        RUN: str = 'run'

    tables = dict(
        metric=SqlLatestMetric,
        param=SqlParam,
        tag=SqlTag,
        run=SqlRun)

    def __init__(self):
        self.listeners = {}  # type: Dict[str, MlflowLoggable]
        self.client = MlflowClient()
        self._engine = None
        self._session = None
        self._conn = None  # for sqlite3 cursor
        self.db_path = 'sqlite:///mlruns.db'
        mlflow.set_tracking_uri(self.db_path)

        # monkeypatch mlflow logging funcs
        mlflow.log_metric = self._log_metric
        mlflow.log_metrics = self._log_metrics
        mlflow.log_param = self._log_param
        mlflow.log_params = self._log_params

        self.reset()

    def reset(self) -> None:
        self._queues = {str(k): deque() for k in list(self.ItemType)}  # type: Dict[str, deque[Dict[str, Any]]]

    def to_dict(self) -> dict:
        return self.num_records

    @property
    def engine(self):
        if self._engine is None:
            self._engine = sa.create_engine(self.db_path)

        return self._engine

    @property
    def session(self) -> 'Session':
        if self._session is None:
            self._session = sessionmaker(bind=self.engine)()

        return self._session

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect('mlruns.db')

        return self._conn

    @property
    def cursor(self):
        """Raw cursor used for db operations other than refreshing main tables"""
        return self.conn.cursor()

    @property
    def num_records(self) -> Dict[str, int]:
        return {k: len(self._queues[k]) for k in self._queues}

    def flush(self) -> None:
        """Flush metric/param queues to db"""
        _backup = copy.deepcopy(self._queues)
        session = self.session
        added = Counter()

        try:
            for item_type, q in self._queues.items():
                while q:
                    m = q.pop()
                    for k, v in m.items():
                        if v is None:
                            log.warning(f'Value for "{k}"={v}')

                    row = self.tables[item_type](**m)
                    session.add(row)
                    added[item_type] += 1

            session.commit()

        except Exception as e:
            session.rollback()
            log.warning('failed load to db')
            self._queues = _backup
            raise e

    def log_item(
            self,
            item_type: str,
            key: str,
            val: Any,
            run_uuid: str = None,
            timestamp: int = None) -> None:

        if not item_type in list(self.ItemType):
            raise ValueError(f'Item must be in {list(self.ItemType)}')

        # init dict record to log to db
        m = dict(
            run_uuid=run_uuid or _get_or_start_run().info.run_id,
            key=key,
            value=val
        )  # type: Dict[str, Any]

        if item_type == self.ItemType.METRIC:
            m['timestamp'] = timestamp or int(time.time() * 1000)
            # is_nan, step?

        # add record to queue
        self._queues[item_type].append(m)

    def _log_metric(self, key: str, val: Any) -> None:
        self.log_item(self.ItemType.METRIC, key, val)

    def _log_metrics(self, items: Dict[str, Any]) -> None:
        for k, v in items.items():
            self._log_metric(k, v)

    def _log_param(self, key: str, val: Any) -> None:
        self.log_item(self.ItemType.PARAM, key, val)

    def _log_params(self, items: Dict[str, Any]) -> None:
        for k, v in items.items():
            self._log_param(k, v)

    def register(self, objs: Union[MlflowLoggable, List[MlflowLoggable]]) -> None:

        for obj in jf.as_list(objs):
            # if not MlflowLoggable in type(obj).__bases__:
            #     raise ValueError(f'object must be instance of {type(MlflowLoggable)}.')

            self.listeners[obj.__class__.__name__.lower()] = obj

    def log_all(self, flush: bool = True) -> None:
        """Log all registered listeners' items"""
        # self.client.log_batch(run.info.run_id, metrics=metrics, params=params, tags=tags)

        for listener in self.listeners.values():
            listener.log_mlflow()

        if flush:
            self.flush()

    @staticmethod
    def log_df(df: pd.DataFrame, name: str = None, keep_index: bool = True) -> None:
        """Save dataframe to feather and log as artifact

        Parameters
        ----------
        df : pd.DataFrame
        name : str, optional
            name, by default temp.ftr
        keep_index : bool, optional
            keep index or drop it (cant save with ftr)
        """
        name = name or 'temp'
        p = cf.p_ftr / f'{name}.ftr'
        df.reset_index(drop=not keep_index).to_feather(p)
        mlflow.log_artifact(p)

    def load_df(self, run: Union['Run', str], name: str = None, index_col: str = None) -> pd.DataFrame:
        """Load feather file from run's artifacts

        Parameters
        ----------
        run : Union[Run, str]
            mlflow Run or run_id
        name : str, optional
            df name, default load first found .ftr file
        index_col : str, optional
            default None

        Returns
        -------
        pd.DataFrame
        """
        if isinstance(run, str):
            run = self.client.get_run(run)

        p = Path(run.info.artifact_uri.replace('file://', ''))

        if name is None:
            ftr_files = [fl.path for fl in self.client.list_artifacts(run.info.run_id) if 'ftr' in fl.path]
            if ftr_files:
                fname = [0]
            else:
                raise FileNotFoundError(f'No feather files found for run: {run.info.run_id}')
        else:
            fname = f'{name}.ftr'

        p = p / fname
        df = pd.read_feather(p)

        # try to get index_col name from predefined names
        index_col = index_col or self.indexes.get(name, None)

        if not index_col is None:
            try:
                df = df.set_index(index_col)
            except:
                log.warning(f'Failed to set index "{index_col}" for "{name}')

        return df

    def collect_dfs(self, name: str, experiment_id: str = '0', **kw) -> list[pd.DataFrame]:
        """Collect dfs from run by name

        Parameters
        ----------
        name : str
            df name eg 'df_pred'
        experiment_id : str, optional
            default '0'

        Returns
        -------
        list[pd.DataFrame]
            list of collected dfs
        """
        dfs = []
        for run in self.client.list_run_infos(experiment_id):
            try:
                df = self.load_df(run=run.run_id, name=name, **kw) \
                    .pipe(pu.append_list, dfs)
            except FileNotFoundError:
                pass

        return dfs

    def compare_df_cols(self, name: str, col_name: str, **kw) -> pd.DataFrame:
        """Collect all dfs by name, concat single column horizontal

        Parameters
        ----------
        name : str
            df name, eg 'df_pred'
        col_name : str
            eg 'rolling_proba'

        Returns
        -------
        pd.DataFrame
            df with single column merged
        """
        dfs = self.collect_dfs(name, **kw)
        return pd.concat([df[col_name].rename(i) for i, df in enumerate(dfs)], axis=1)

    def check_end(self) -> None:
        """Check run ended correctly in case of errors"""
        run = mlflow.active_run()
        if run and run.info.lifecycle_stage == 'active':
            try:
                mlflow.end_run()
                mlflow.delete_run(run.info.run_id)
            except:
                log.warning('Failed to delete run')

    def add_prev_item(self, key: str, val: Any, table: str = 'metric') -> None:
        """Add metric for all runs which wasn't previously logged

        Parameters
        ----------
        key : str
            metric key
        val : Any
            value to add
        """
        name = dict(
            metric='latest_metrics',
            param='params')[table]

        df = self.df_results() \
            .reset_index(drop=False)[['run_id']] \
            .rename(columns=dict(run_id='run_uuid'))

        data = dict(key=key, value=val)

        if table == 'metric':
            data |= dict(step=0, is_nan=0)

        df \
            .assign(**data) \
            .to_sql(name=name, con=self.engine, if_exists='append', index=False)

        log.info(f'Added {table} "{key}={val}" for {len(df)} runs to table "{name}".')

    def df_results(self, experiment_ids: Union[str, List[str]] = '0', by: str = 'ci_monthly') -> pd.DataFrame:

        int_cols = ['target_n_periods', 'n_periods_smooth', 'weights_n_periods',
                    'n_estimators', 'num_leaves', 'max_depth', 'num_feats']
        cols2 = ['ci_monthly', 'final', 'w_sharpe', 'sharpe', 'filter_fit_quantile',
                 'order_offset'] + int_cols + ['w_acc', 'acc', 'tpd', 'drawdown']

        return mlflow.search_runs(experiment_ids) \
            .dropna() \
            .set_index(['experiment_id', 'run_id']) \
            .pipe(lambda df: df[[c for c in df.columns if re.match(r'metr|para', c)]]) \
            .pipe(lambda df: df.rename(columns={c: c.split('.')[1] for c in df.columns})) \
            .pipe(pu.convert_dtypes, cols=int_cols, _type=int) \
            .sort_values(by=by, ascending=False) \
            .pipe(pu.reorder_cols, cols=cols2) \
            .round(dict(filter_fit_quantile=3, order_offset=5)) \
            .reset_index(drop=True)

    def pairplot(self, df: pd.DataFrame = None, latest: bool = False, experiment_ids: str = '0', **kw) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if df is None:
            df = self.df_results(experiment_ids=experiment_ids)

        if latest:
            df = df.loc[df.end == df.end.max()]
            print(len(df))

        df = df.assign(end=lambda df: df.end.astype(str).astype(pd.CategoricalDtype()))

        x_vars = [
            'target_n_periods',
            'n_periods_smooth',
            'filter_fit_quantile',
            'tpd',
            'n_estimators',
            'max_depth',
            'num_leaves',
            'num_feats',
            'batch_size'
        ]

        y_vars = ['w_acc', 'ci_monthly', 'sharpe', 'w_sharpe']

        fig, axs = plt.subplots(
            nrows=len(x_vars),
            ncols=len(y_vars),
            figsize=(3 * len(x_vars), 2.5 * 6 * len(y_vars))
        )

        for xi, xcol in enumerate(x_vars):
            for yi, ycol in enumerate(y_vars):

                sns.scatterplot(
                    y=df[ycol],
                    x=df[xcol],
                    hue=df.end,
                    ax=axs[xi, yi],
                    legend=False,
                    alpha=0.4,
                    linewidth=0)

        plt.autoscale(True)
        plt.tight_layout(pad=0.4)


def import_local(mfm: MlflowManager, kill: bool = False) -> None:

    _p = Path('mlruns/0')
    runs = 0
    mfm.reset()

    for p in _p.iterdir():

        if not p.is_dir():
            continue

        run_uuid = p.name
        p_meta = p / 'meta.yaml'
        if not p_meta.exists():
            log.warning(f'Skipping: {p_meta}')
            continue

        mfm._queues[mfm.ItemType.RUN].append(load_meta(p_meta))

        if kill:
            p_meta.unlink()

        # metrics, params, tags
        for item_type in ('metric', 'param', 'tag'):

            for p_item in (p / f'{item_type}s').iterdir():

                with open(p_item, 'r') as file:
                    _val = file.readline().rstrip()

                if item_type == 'metric':
                    timestamp, val, is_nan = _val.split()
                else:
                    timestamp = 0
                    val = _val

                mfm.log_item(item_type, p_item.name, val, run_uuid=run_uuid, timestamp=int(timestamp))

                if kill:
                    p_item.unlink()

        runs += 1

    log.info(f'processed [{runs}] runs')


def load_meta(p: Path) -> Dict[str, Any]:

    with open(p, 'r') as file:
        m = yaml.full_load(file)

    m['source_type'] = SourceType.to_string(m['source_type'])
    m['status'] = RunStatus.to_string(m['status'])

    exclude = ('run_id', 'tags')
    for k in exclude:
        del m[k]

    return m


def convert_bytes(val) -> int:
    if isinstance(val, bytes):
        return struct.unpack('ii', val)[0]
    elif isinstance(val, str):
        try:
            return int(val)
        except:
            try:
                return float(val)
            except:
                return val
    else:
        return val


def param_to_metric(df: pd.DataFrame, mfm: MlflowManager, cols: List[str] = None):
    cols = cols or ['n_periods_smooth']
    # .assign(n_periods_smooth=lambda df: df['n_periods_smooth'].apply(convert_bytes)) \

    df2 = df[cols].reset_index(drop=False) \
        .drop(columns=['experiment_id']) \
        .rename(columns=dict(run_id='run_uuid')) \
        .melt(id_vars=['run_uuid'], var_name='key') \
        .assign(step=0, timestamp=0, is_nan=0)

    df2.to_sql(name='latest_metrics', con=mfm.conn, if_exists='append', index=False)

    sql = f"""
    delete from params where key in {tuple(cols)}
    """
    mfm.cursor.execute(sql)
    mfm.conn.commit()

    return df2
