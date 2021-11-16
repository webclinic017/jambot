from abc import ABCMeta, abstractproperty
from typing import *

import mlflow
import pandas as pd

from jambot import config as cf
from jambot import functions as f


class MLFlowLoggable(object, metaclass=ABCMeta):
    """Class to ensure object provides params/metrics for mlflow logging"""

    def register(self, mfm: 'MLFlowManager') -> None:
        """Attach self to mf manager

        Parameters
        ----------
        mfm : MLFlowManager

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
        """Log params + metrics for self and strategy to mlflow"""
        params, metrics = self.get_metrics_params()

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        if hasattr(self, 'log_artifact'):
            mlflow.log_artifact(self.log_artifact)


class MLFlowManager():
    def __init__(self):
        self.listeners = []  # type: List[MLFlowLoggable]

    def register(self, objs: Union[MLFlowLoggable, List[MLFlowLoggable]]) -> None:

        for obj in f.as_list(objs):
            # if not MLFlowLoggable in type(obj).__bases__:
            #     raise ValueError(f'object must be instance of {type(MLFlowLoggable)}.')

            self.listeners.append(obj)

    def log_all(self) -> None:
        """Log all registered listeners' items"""
        for listener in self.listeners:
            listener.log_mlflow()

    def log_df(self, df: pd.DataFrame, name: str = None, keep_index: bool = True) -> None:
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

    def check_end(self) -> None:
        """Check run ended correctly in case of errors"""
        run = mlflow.active_run()
        if run and run.info.lifecycle_stage == 'active':
            mlflow.end_run()
            mlflow.delete_run(run.info.run_id)
