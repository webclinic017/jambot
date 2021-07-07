from datetime import datetime as dt
from datetime import timedelta as delta

import pandas as pd

from jambot import SYMBOL
from jambot import functions as f
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.common import DictRepr
from jambot.database import db
from jambot.ml import models as md

log = getlog(__name__)


class ModelStorageManager(DictRepr):
    """Class to manage saving/training/loading ml models"""

    def __init__(
            self,
            batch_size: int = 24,
            n_models: int = 3,
            d_lower: dt = None):

        dt_format = '%Y-%m-%d-%H'
        p_model = f.p_data / 'models'
        f.check_path(p_model)

        if d_lower is None:
            d_lower = dt(2017, 1, 1)

        f.set_self(vars())

    def clean(self) -> None:
        """Clean all saved models in models dir"""
        for p in self.p_model.glob('*'):
            p.unlink()

    def fit_save(self, df: pd.DataFrame, name: str, estimator) -> None:
        """fit model and save

        - all models start fit from same d_lower
        - new versions created for each new saved ver

        Parameters
        ----------
        estimator :
            estimator with fit/predict methods
        d_upper : dt
            [description]
        """
        cfg = md.get_model_params(name)

        for i in range(self.n_models):

            # d = self.d_upper + delta(hours=-i * self.batch_size)

            # filter df to max date
            df = self.df.iloc[:(-i * self.batch_size) or None]

            d = df.index[-1]

            # fit
            x_train, y_train = sk.split(df, target=cfg['target'])
            estimator.fit(x_train, y_train)

            # save
            fname = f'{name}_{d:self.dt_format}'
            f.save_pickle(estimator, p=self.p_model, name=fname)
            log.info(f'saved model: {fname}, max_date: {d}')

    def get_preds(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Load all models, make iterative predictions

        Parameters
        ----------
        name : str
            model name

        Returns
        -------
        pd.DataFrame
            df with all predictions added
        """
        cfg = md.get_model_params(name)
        target = cfg['target']
        dfs = []

        # d_max = df.index[-1]
        p_models = sorted(self.p_model.glob(f'*{name}*'))

        # get date from filename
        # dates = [dt.strptime(p.stem.split('_')[-1], self.dt_format) for p in p_models]

        for i, p in enumerate(p_models):

            d = dt.strptime(p.stem.split('_')[-1], self.dt_format)

            # load estimator
            pipe = f.load_pickle(p)

            if not i + 1 == len(p_models):
                max_date = d + delta(hours=self.batch_size)
            else:
                max_date = df.index[-1]

            x, _ = sk.split(df.loc[d: max_date], target=target)

            df_pred = pd.DataFrame(
                data=pipe.predict(x),
                columns=target,
                index=x.index)

            log.info(f'Adding preds: {len(df_pred)}')

            dfs.append(df_pred)

        return df.pipe(f.left_merge, pd.concat(dfs))

    def to_dict(self):
        return ('d_lower', 'd_upper', 'n_models', 'batch_size')


def load_df(d_lower: dt, symbol: str = 'XBTUSD') -> pd.DataFrame:
    """Load base df from database"""
    kw = dict(
        symbol=symbol,
        startdate=d_lower,
        interval=1)

    return db.get_dataframe(**kw) \
        .drop(columns=['Timestamp', 'Symbol'])


def fit_save_models(df: pd.DataFrame = None, d_lower: dt = None, reset_hour: int = 18):
    """Fit lgbm and Ridge models for current strategy predictions

    reset_hour = 11 pst = 18 utc
    """
    n_periods = 10  # wet

    if d_lower is None:
        d_lower = dt(2017, 1, 1)

    # filter to constant d_upper per day
    d_upper = f.date_to_dt(dt.utcnow().date()) + delta(hours=reset_hour)

    if df is None:
        df = load_df(d_lower=d_lower, symbol=SYMBOL)

    # drop last rows which we cant set proper target
    df = df.copy() \
        .loc[:d_upper] \
        .pipe(md.add_signals, name='lgbm') \
        .pipe(md.add_signals, name='ridge') \
        .iloc[:-1 * n_periods, :] \
        .fillna(0)

    msm = ModelStorageManager()
    msm.clean()

    names = ('lgbm', 'ridge')

    for name in names:
        pipe = md.make_pipeline(name=name, df=df)
        msm.fit_save(df=df, name=name, estimator=pipe)
