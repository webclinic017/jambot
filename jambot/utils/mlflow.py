import copy
import re
import time
from abc import ABCMeta, abstractproperty
from collections import Counter, deque
from pathlib import Path
from typing import *

import mlflow
import pandas as pd
import sqlalchemy as sa
import yaml
from aenum import StrEnum
from mlflow.entities.run import Run
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType
from mlflow.store.tracking.dbmodels.models import (SqlLatestMetric, SqlParam,
                                                   SqlRun, SqlTag)
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_or_start_run
from sqlalchemy.orm import Session, sessionmaker

from jambot import config as cf
from jambot import getlog
from jambot.common import DictRepr
from jgutils import functions as jf
from jgutils import pandas_utils as pu

log = getlog(__name__)


class MlflowLoggable(object, metaclass=ABCMeta):
    """Class to ensure object provides params/metrics for mlflow logging"""

    def register(self, mfm: 'MlflowManager') -> 'MlflowLoggable':
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
            if isinstance(v, (float, int)) and not isinstance(v, bool):
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
        if hasattr(self, 'log_artifact'):
            mlflow.log_artifact(self.log_artifact())

        if hasattr(self, 'log_dfs'):
            for kw in jf.as_list(self.log_dfs()):
                MlflowManager.log_df(**kw)


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
        self.db_path = 'sqlite:///mlruns.db'

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
                    row = self.tables[item_type](**m)
                    session.add(row)
                    added[item_type] += 1

            session.commit()
            log.info(f'Added rows to db: {dict(added)}')

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

    def load_df(self, run: Union[Run, str], name: str = None, index_col: str = None) -> pd.DataFrame:
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

    def df_results(self, experiment_ids: Union[str, List[str]] = '0') -> pd.DataFrame:

        return mlflow.search_runs(experiment_ids) \
            .set_index(['experiment_id', 'run_id']) \
            .sort_values('start_time') \
            .pipe(lambda df: df[[c for c in df.columns if re.match(r'metr|para', c)]]) \
            .pipe(lambda df: df.rename(columns={c: c.split('.')[1] for c in df.columns})) \
            .drop(columns=['start', 'end'])  # temp for displaying


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
