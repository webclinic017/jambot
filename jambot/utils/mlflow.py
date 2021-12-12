import re
from abc import ABCMeta, abstractproperty
from pathlib import Path
from typing import *

import mlflow
import pandas as pd
from jgutils import functions as jf
from jgutils import pandas_utils as pu
from mlflow.entities.run import Run
from mlflow.tracking import MlflowClient

from jambot import config as cf
from jambot import getlog

log = getlog(__name__)


class MlflowLoggable(object, metaclass=ABCMeta):
    """Class to ensure object provides params/metrics for mlflow logging"""

    def register(self, mfm: 'MlflowManager') -> None:
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


class MlflowManager():
    indexes = dict(
        df_pred='timestamp',
        df_trades='timestamp')

    def __init__(self):
        self.listeners = {}  # type: Dict[str, MlflowLoggable]
        self.client = MlflowClient()

    def register(self, objs: Union[MlflowLoggable, List[MlflowLoggable]]) -> None:

        for obj in jf.as_list(objs):
            # if not MlflowLoggable in type(obj).__bases__:
            #     raise ValueError(f'object must be instance of {type(MlflowLoggable)}.')

            self.listeners[obj.__class__.__name__.lower()] = obj

    def log_all(self) -> None:
        """Log all registered listeners' items"""
        for listener in self.listeners.values():
            listener.log_mlflow()

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
