from typing import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from jambot import getlog
from jambot.common import DictRepr, DynConfig
from jambot.utils.mlflow import MlflowLoggable
from jgutils import functions as jf

log = getlog(__name__)


class WeightsManager(DictRepr, MlflowLoggable, DynConfig):
    """Class to manage creating weighted series and filtering by weights"""
    log_keys = dict(weights_n_periods='n_periods')

    def __init__(self, df: pd.DataFrame, n_periods: int, weight_linear: bool = True):

        # weight history linearly if live training
        self.linear = (lambda x: np.linspace(0.5, 1, len(x))) if weight_linear else lambda x: 1.0

        cols = ['close']
        self.df = df[cols]
        self.n_periods = n_periods
        self.weight_linear = weight_linear
        self._weights = None

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
                    X=x[target_cols].abs().max(axis=1)
                    .clip(upper=0.2) * self.linear(x),
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
            s = s.pipe(self.filter_quantile, weights=s, quantile=filter_quantile)

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

    def filter_quantile(
            self,
            datas: Union[pd.DataFrame, pd.Series, List[Union[pd.DataFrame, pd.Series]]],
            quantile: float,
            weights_in_frame: bool = False,
            _log: bool = True) -> Union[pd.DataFrame, pd.Series, list]:
        """Filter df or series to highest qualtile based on index of weights
        - IMPORTANT: higher number = more rows filtered out, less remain
        - only filters based on quantile of weights matching given index

        Parameters
        ----------
        datas : Union[pd.DataFrame, pd.Series, List[Union[pd.DataFrame, pd.Series]]]
            single or list of df/series
        quantile : float
            filter above this quantile

        Returns
        -------
        Union[pd.DataFrame, pd.Series, List[Union[pd.DataFrame, pd.Series]]]
            df or series filtered by weight quantile, or list of both
        """
        out = []
        datas = jf.as_list(datas)
        for df in datas:
            # filter weights UP TO MAX DATE of df.index before getting quantile
            if not weights_in_frame:

                # get max date in df timestamp index (works single or multiindex)
                d_max = df.index.get_level_values('timestamp').max()

                if 'symbol' in df.index.names:
                    idx_slice = pd.IndexSlice[:, :d_max]  # multiindex (symbol, timestamp)
                else:
                    idx_slice = pd.IndexSlice[:d_max]  # timestamp only

                weights_all = self.weights.loc[idx_slice]
                weights_current = self.weights.loc[df.index]
            else:
                # using dask df, need to do things the hard way (not used)
                weights_all = df['weights'].compute()
                weights_current = weights_all

            # get value of weights equal to quantile eg q=0.6 > qv=0.0564
            quantile_val = weights_all.quantile(quantile)

            if not weights_in_frame:
                idx = weights_current[weights_current >= quantile_val].index
                out.append(df.loc[idx])
            else:
                out.append(df[df.weights >= quantile_val])

        if _log:
            nrows = datas[0].shape[0]
            msg = f'Filtered weights quantile [{quantile * 100:.0f}%, qv={quantile_val:.3f}]' \
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
