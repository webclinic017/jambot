from typing import *

import numpy as np
import pandas as pd
from jgutils import functions as jf
from sklearn.preprocessing import minmax_scale

from jambot import config as cf
from jambot import getlog
from jambot.common import DictRepr
from jambot.utils.mlflow import MlflowLoggable

log = getlog(__name__)


class WeightsManager(DictRepr, MlflowLoggable):
    """Class to manage creating weighted series and filtering by weights"""

    def __init__(self, df: pd.DataFrame, n_periods: int, weight_linear: bool = True):

        # weight history linearly if live training
        self.linear = (lambda x: np.linspace(0.5, 1, len(x))) if weight_linear else lambda x: 1.0

        cols = ['high', 'low', 'close']
        self.df = df[cols]
        self.n_periods = n_periods
        self.weight_linear = weight_linear
        self._weights = None

    @classmethod
    def from_config(cls, df: pd.DataFrame) -> 'WeightsManager':
        return cls(df=df, **cf.config['weightsmanager_kw'])

    def to_dict(self) -> dict:
        return dict(
            df=f'{self.df.shape[0]:,.0f}',
            n_periods=self.n_periods,
            weight_linear=self.weight_linear)

    @property
    def log_items(self) -> dict:
        return dict(
            weights_n_periods=self.n_periods,
            weight_linear=self.weight_linear)

    @property
    def weights(self) -> pd.Series:
        if self._weights is None:
            self._weights = self._get_weights()

        return self._weights

    def _get_weights(self) -> pd.Series:
        """Set base weights series
        """
        from jambot.signals import TargetMaxMin

        target_cols = ['target_max', 'target_min']

        # get max/min of close in n periods
        signals = TargetMaxMin(n_periods=self.n_periods, use_close=True).signals

        signals = {k: v['func'] for k, v in signals.items()} \
            | dict(
                weight=lambda x: minmax_scale(
                    X=x[target_cols].abs().max(axis=1).clip(upper=0.2) * self.linear(x),
                    feature_range=(0, 1)))

        return self.df \
            .assign(**signals) \
            .assign(weight=lambda x: x.weight.fillna(x.weight.mean()).astype(np.float32))['weight']

    def get_weight(self, df: pd.DataFrame, filter_quantile: int = None) -> pd.Series:
        """return single array of weighted values to pass as fit_params

        Parameters
        ----------
        df : pd.DataFrame
            df with ['high', 'low', 'close']

        Returns
        -------
        pd.Series
            column weighted by abs movement up/down in next n_periods
        """

        s = self.weights
        if not filter_quantile is None:
            s = s.pipe(self.filter_highest, weights=s, quantile=filter_quantile)

        return s

    def fit_params(self, x: pd.DataFrame, name: str = None) -> Dict[str, pd.Series]:
        """Create dict of weighted samples for fit params

        Parameters
        ----------
        x : pd.DataFrame
        name : str, optional
            model name to prepend to dict key, by default None

        Returns
        -------
        Dict[str, pd.Series]
            {lgbm__sample_weight: [0.5, ..., 1.0]}
        """
        name = f'{name}__' if not name is None else ''
        return {f'{name}sample_weight': self.weights.loc[x.index]}

    def filter_highest(
            self,
            datas: Union[pd.DataFrame, pd.Series, list],
            quantile: float = 0.5) -> Union[pd.DataFrame, pd.Series, list]:
        """Filter df or series to highest qualtile based on index of weights

        Parameters
        ----------
        datas : Union[pd.DataFrame, pd.Series, list]
            single or list of df/series
        quantile : float, optional
            filter above this quantile, default 0.5

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
            df or series filtered by weight quantile, or list of both
        """

        out = []
        datas = jf.as_list(datas)
        for df in datas:
            # filter weights to only df.index before getting quantile
            weights = self.weights.loc[df.index]

            # if not quantile is None:
            _quantile = weights.quantile(quantile)

            idx = weights[weights >= _quantile].index
            out.append(df.loc[idx])

        nrows = datas[0].shape[0]
        msg = f'Filtered weights quantile [{quantile * 100:.0f}% = {_quantile:.3f}]' \
            + f', [{nrows:,.0f} -> {idx.shape[0]:,.0f}] rows.'
        log.info(msg)

        if len(datas) == 1:
            return out[0]
        else:
            return out

    def show_plot(self, weight: pd.Series = None) -> None:
        """Show scatter plot of dist of weights

        Parameters
        ----------
        weight : pd.Series, optional
            from self.get_weight(), by default None
        """
        if weight is None:
            weight = self.weights

        weight.to_frame() \
            .reset_index(drop=False) \
            .plot(kind='scatter', x='timestamp', y='weight', s=1, alpha=0.1)
