from typing import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot.common import DictRepr

log = getlog(__name__)


class WeightsManager(DictRepr):
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
    def weights(self) -> pd.Series:
        if self._weights is None:
            self._set_weights()

        return self._weights

    def _set_weights(self) -> None:
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

        self._weights = self.df \
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
        # cols = ['high', 'low', 'close']
        # s = df[cols] \
        #     .pipe(self.add_all_signals) \
        #     .assign(weight=lambda x: x.weight.fillna(x.weight.mean()).astype(np.float32))['weight']

        s = self.weights
        if not filter_quantile is None:
            s = s.pipe(self.filter_highest, weights=s, quantile=filter_quantile)

        return s

    def fit_params(self, x: pd.DataFrame, name: str = None) -> Dict[str, np.ndarray]:
        """Create dict of weighted samples for fit params

        Parameters
        ----------
        name : str, optional
            model name to prepend to dict key, by default None

        Returns
        -------
        Dict[str, np.ndarray]
            {lgbm__sample_weight: [0.5, ..., 1.0]}
        """
        name = f'{name}__' if not name is None else ''

        # if weights is None:
        #     weights = np.linspace(0.5, 1, n)

        return {f'{name}sample_weight': self.weights.loc[x.index]}

    def filter_highest(
            self,
            datas: Union[pd.DataFrame, pd.Series, list],
            quantile: float = 0.5,
    ) -> Union[pd.DataFrame, pd.Series, list]:
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
        datas = f.as_list(datas)
        for df in datas:
            # filter weights to only df.index before getting quantile
            weights = self.weights.loc[df.index]

            # if not quantile is None:
            _quantile = weights.quantile(quantile)

            idx = weights[weights >= _quantile].index
            out.append(df.loc[idx])

        nrows = df.shape[0]
        msg = f'Filtered weights quantile [{quantile * 100:.0f}% = {_quantile:.3f}]' \
            + f', [{nrows:,.0f} -> {idx.shape[0]:,.0f}] rows.'
        log.info(msg)

        if len(datas) == 1:
            return out[0]
        else:
            return out

    def show_plot(self, weight: pd.Series = None, df: pd.DataFrame = None) -> None:
        """Show scatter plot of dist of weights

        Parameters
        ----------
        weight : pd.Series, optional
            from self.get_weight(), by default None
        df : pd.DataFrame, optional
        """
        if weight is None:
            weight = self.get_weight(df)

        weight.to_frame() \
            .reset_index(drop=False) \
            .plot(kind='scatter', x='timestamp', y='weight', s=1, alpha=0.1)
