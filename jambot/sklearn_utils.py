import copy
import re
import time
from datetime import datetime as dt
from typing import *

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from jambot import config as cf
from jambot import display
from jambot import functions as f
from jambot import getlog
from jambot import signals as sg
from jambot.common import ProgressParallel
from jambot.ml import models as md
from jambot.utils.azureblob import BlobStorage

# not needed when running on azure
try:
    import shap
    from icecream import ic
    ic.configureOutput(prefix='')

    import matplotlib.pyplot as plt

except (ImportError, ModuleNotFoundError) as e:
    pass

log = getlog(__name__)


class ModelManager(object):
    """Manager class to perform cross val etc on multiple models with same underlying column transformer + data
    """

    def __init__(
            self,
            ct=None,
            scoring=None,
            cv_args=None,
            random_state=0,
            target: str = 'target',
            scorer=None,
            **kw):
        cv_args = cv_args if not cv_args is None else {}
        df_results = pd.DataFrame()
        pipes = {}
        scores = {}
        models = {}
        grids = {}
        df_preds = {}
        cv_data = {}  # used to return estimator/strats for each cv fold
        v = {**vars(), **kw}
        f.set_self(v)

        if any(item in kw for item in ('features', 'encoders')):
            self.make_column_transformer(**kw)

    def init_cv(self, scoring: dict, cv_args: dict, scorer=None) -> 'ModelManager':
        """Convenience func to make sure all reqd cross val params are set

        Parameters
        ----------
        scoring : dict
            scoring names/funcs
        cv_args : dict
            cross validation args dict
        scorer : [type], optional
            jambot.tradesys.strategies.StratScorer, by default None
        """
        f.set_self(vars())
        return self

    def make_column_transformer(self, features: dict, encoders: dict, **kw) -> ColumnTransformer:
        """Create ColumnTransformer from dicts of features and encoders
        - NOTE column for CountVectorizer must be 'column_name' not ['column_name']

        Parameters
        ----------
        features : dict
            feature group names matched to columns
        encoders : dict
            feature group names matched to encoders, eg MinMaxScaler

        Returns
        -------
        ColumnTransformer
        """
        ct = ColumnTransformer(
            transformers=[(name, encoder, features[name]) for name, encoder in encoders.items()])

        f.set_self(vars())
        return ct

    def show_ct(self, x_train: pd.DataFrame = None):
        """Show dataframe summary of transformed columns
        - NOTE doesn't work with CountVectorizer (could be thousands of colums), just skip calling this"""

        if x_train is None:
            x_train = getattr(self, 'x_train', None)

        if x_train is None:
            raise AttributeError('x_train not set!')

        data = self.ct.fit_transform(x_train)
        df_trans = df_transformed(data=data, ct=self.ct)
        log.info(df_trans.shape)

        with pd.option_context('display.max_rows', 200):
            display(df_trans.describe().T)

    def get_model(self, name: str, best_est=False):
        if best_est:
            return self.best_est(name=name)
        else:
            return self.models[name]

    def cross_val_feature_params(
            self,
            signal_manager,
            name,
            model,
            feature_params: dict,
            train_size: float = 0.8) -> None:
        """Run full cross val pipe with single replacement of each feature in feature_params"""
        df_scores_features = pd.DataFrame()

        for df, param_name in signal_manager \
            .replace_single_feature(
                df=self.df,
                feature_params=feature_params):

            # need to remake train/test splits every time
            x_train, y_train, x_test, y_test = self.make_train_test(
                df=df,
                target=self.target,
                train_size=train_size,
                shuffle=False)

            # just remake models dict with modified param name as key
            models = {f'{name}_{param_name}': model}
            self.cross_val(models=models, show=False, df_scores=df_scores_features)

        self.show(df=df_scores_features)

    def timeit(self, func, *args, **kw):
        t = time.time()
        res = func(*args, **kw)
        final_time = time.time() - t
        return final_time, res

    def fit_score(self, models: dict, show: bool = True):
        """Simple timed fit/score into df to display results"""
        for name, model in models.items():

            # allow passing model definition, or instantiated model
            if isinstance(model, type):
                model = model()

            t = time.time()
            model.fit(self.x_train, self.y_train)

            fit_time, _ = self.timeit(model.fit, self.x_train, self.y_train)
            score_time, train_score = self.timeit(
                model.score, self.x_train, self.y_train)

            scores = dict(
                fit_time=fit_time,
                score_time=score_time,
                train_score=train_score,
                test_score=model.score(self.x_test, self.y_test))

            self.scores[name] = scores
            self.models[name] = model
            self.df_results.loc[name, scores.keys()] = scores.values()

        if show:
            self.show()

    def make_pipe(self, name: str, model, steps: list = None) -> Pipeline:
        pipe = Pipeline(
            steps=[
                ('ct', self.ct),
                (name, model)],
        )

        # insert extra steps in pipe, eg RFECV
        if not steps is None:
            for step in f.as_list(steps):
                pipe.steps.insert(step[0], step[1])

        self.pipes[name] = pipe

        return pipe

    def init_models(self, models: Dict[str, BaseEstimator], **kw) -> None:
        """Convenience func to init dict of models

        Parameters
        ----------
        models : Dict[str, BaseEstimator]
        """
        for name, model in models.items():
            pipe = self.init_model(name=name, model=model, **kw)

    def init_model(
            self,
            name: str,
            model: BaseEstimator,
            steps: List[Tuple[int, tuple]] = None) -> Pipeline:
        """Init single model

        Parameters
        ----------
        name : str
            model name
        model : BaseEstimator
        steps : List[Tuple[int, tuple]], optional
            default None

        Returns
        -------
        Pipeline
            initialized Pipeline with ColumnTransformer
        """

        # allow passing model definition, or instantiated model
        if isinstance(model, type):
            model = model()

        model.random_state = self.random_state
        self.models[name] = model
        return self.make_pipe(name=name, model=model, steps=steps)

    def cross_val(
            self,
            models: dict,
            show: bool = True,
            steps: List[Tuple[int, tuple]] = None,
            df_scores: pd.DataFrame = None,
            extra_cv_args: dict = None):
        """Perform cross validation on multiple classifiers

        Parameters
        ----------
        models : dict
            models with {name: classifier} to cross val
        show : bool, optional
            show dataframe of results, default True
        steps : list, optional
            list of tuples of [(step_pos, (name, model)), ]
        """
        if self.ct is None:
            raise AttributeError('ColumnTransformer not init!')

        if df_scores is None:
            df_scores = self.df_results

        # reset StratScorer before each cross_val
        if not self.scorer is None:
            self.scorer.reset()

        for name, model in models.items():
            pipe = self.init_model(name=name, model=model, steps=steps)

            if extra_cv_args:
                self.cv_args |= extra_cv_args

            scores = cross_validate(
                pipe, self.x_train, self.y_train, error_score='raise', **self.cv_args)

            # use estimator obj to return dfs from cross_val for plotting
            if 'return_estimator' in self.cv_args:
                self.cv_data[name] = scores['estimator']
                scores = {k: v for k, v in scores.items() if not k == 'estimator'}

            self.scores[name] = scores
            df_scores = df_scores \
                .pipe(
                    append_mean_std_score,
                    scores=scores,
                    name=name,
                    scoring=self.cv_args.get('scoring', None))

        if show:
            self.show()

    def show(self, df=None):
        if df is None:
            df = self.df_results

        show_scores(df)

    def fit(
            self,
            model: BaseEstimator,
            fit_params: dict = None,
    ) -> BaseEstimator:
        """Fit model to self.x_train with self.y_train (if not fit already)

        Parameters
        ----------
        model : BaseEstimator
            estimator/pipeline
        fit_params : dict, optional
            eg lgbm__sample_weight, by default None

        Returns
        -------
        BaseEstimator
            fitted model
        """
        fit_params = fit_params or {}

        if not is_fitted(model):
            log.info('fitting model')
            model.fit(self.x_train, self.y_train, **fit_params)

        return model

    def y_pred(self, x: pd.DataFrame, model=None, **kw):
        # if model is None or not is_fitted(model):
        model = self.fit(model=model, **kw)

        return model.predict(x)

    def class_rep(self, name: str = None, **kw):
        """Show classification report

        Parameters
        ----------
        name : str
            name of existing model
        """
        y_pred = self.y_pred(name=name, X=self.x_test, **kw)
        y_true = self.y_test.values.flatten()

        # classification report
        m = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(m).T
        display(df)

    def add_predict_iter(
            self,
            df: pd.DataFrame,
            name: str,
            model: BaseEstimator = None,
            batch_size: int = 96,
            min_size: int = 180 * 24,
            max_train_size: int = None,
            regression: bool = False) -> pd.DataFrame:
        """Retrain model every x periods and add predictions for next batch_size"""
        from lightgbm import LGBMClassifier

        nrows = df.shape[0]
        num_batches = ((nrows - min_size) // batch_size) + 1

        cfg = md.model_cfg(name)
        weights = sg.WeightedPercentMaxMin(
            cfg['n_periods_weighted'],
            weight_linear=True).get_weight(df)

        if model is None:
            model = self.pipes[name].named_steps[name]

        drop_cols = [c for c in cf.config['drop_cols'] if c in df.columns]
        df_ohlc = df[drop_cols]
        df = df.pipe(f.safe_drop, drop_cols)

        def _get_imp_feats(x, y):
            """Fit shap on history up to this point
            - use n most important cols going forward
            - NOTE could limit fitting important cols up to eg last year?
            """
            # need to pass in x/y for history till now
            # need to know if we need ct for column names or not
            spm = ShapManager(
                x=x,
                y=y,
                ct=None,
                model=LGBMClassifier(num_leaves=50, n_estimators=50, max_depth=30,
                                     boosting_type='dart', learning_rate=0.1),
                # model=self.models['lgbm'],
                weights=weights,
                n_sample=10_000)

            return spm.shap_n_important(n=60, save=False, upload=False, as_list=True)['most']

        def _fit(i: int) -> pd.DataFrame:
            i_lower = min_size + i * batch_size  # "test" lower
            i_upper = min(i_lower + batch_size, nrows + 1)
            idx_test = df.index[i_lower: i_upper]

            # train model from 0 up to current position
            # NOTE max_train_size eg 2 yrs seems to be much worse
            i_train_lower = 0 if not max_train_size else max(0, i_lower - max_train_size)

            # if cols is None:
            #     cols = df.columns

            x_train, y_train = split(
                df.iloc[i_train_lower: i_lower],
                target=self.target)

            # use shap to get important cols at each iter
            # imp_cols = _get_imp_feats(x_train, y_train)
            # x_train = x_train[imp_cols]

            model.fit(
                x_train,
                y_train,
                sample_weight=weights.loc[x_train.index]
                # **weighted_fit(name=None, weights=weights.loc[x_train.index])
            )

            x_test, _ = split(df.loc[idx_test], target=self.target)
            # x_test = x_test[imp_cols]

            if len(idx_test) == 0:
                return None
            else:
                if not regression:
                    proba_long = df_proba(x=x_test, model=model)['proba_long']
                else:
                    proba_long = df_y_pred(x=x_test, model=model)['y_pred']

                return pd.DataFrame(
                    index=idx_test,
                    data=dict(
                        y_pred=model.predict(x_test),
                        proba_long=proba_long))

        result = ProgressParallel(n_jobs=-1, total=num_batches)(delayed(_fit)(i=i) for i in range(num_batches))
        return df_ohlc.pipe(f.left_merge, pd.concat([df for df in result if not df is None]))

    def add_predict(
            self,
            df: pd.DataFrame,
            x_test: pd.DataFrame = None,
            proba: bool = True,
            model: BaseEstimator = None,
            name: str = None,
            fit_params: dict = None,
            **kw) -> pd.DataFrame:
        """Add predicted vals to df

        Parameters
        ----------
        df : pd.DataFrame
            df with signals added
        proba : bool, optional
            add predicted probabilites as well as predictions, default True

        Returns
        -------
        pd.DataFrame
            df with predictions and optionally probabilities
        """
        if model is None:
            model = self.pipes[name]

        model = self.fit(model, fit_params=fit_params)

        if x_test is None:
            x_test = self.x_test

        df = df \
            .pipe(add_y_pred, model=model, x=x_test) \
            .pipe(add_probas, model=model, x=x_test, do=proba)

        # save predicted values for each model
        self.df_preds[name] = df

        return df

    def add_predict_multi(self, df, model):
        """Add multiple prediction columns to df"""

        return

    def best_est(self, name: str):
        return self.grids[name].best_estimator_

    def search(self, name: str, params: dict, estimator=None, search_type: str = 'random', **kw):
        """Perform Random or Grid search to optimize params on specific model

        Parameters
        ----------
        name : str
            name of model, must have been previously added to ModelManager
        params : dict
            [description]
        estimator : sklearn model/Pipeline, optional
            pass in model if not init already, default None
        search_type : str, optional
            RandomSearchCV or GridSearchCV, default 'random'

        Returns
        -------
        RandomSearchCV | GridSearchCV
            sklearn model_selection object
        """
        # TODO need to enable NO renaming

        if estimator is None:
            estimator = self.pipes[name]
            model = estimator.named_steps[name]
        else:
            model = estimator

        # MultiOutputRegressor needs estimator__
        if isinstance(model, MultiOutputRegressor):
            log.info('multioutput')
            params = {f'estimator__{k}': v for k, v in params.items()}

        # rename params to include 'name__parameter'
        params = {f'{name}__{k}': v for k, v in params.items()}

        m = dict(
            random=dict(
                cls=RandomizedSearchCV,
                param_name='param_distributions'),
            grid=dict(
                cls=GridSearchCV,
                param_name='param_grid')) \
            .get(search_type)

        # grid/random have different kw for param grid/distribution
        kw[m['param_name']] = params

        grid = m['cls'](
            estimator=estimator,
            **kw,
            **self.cv_args) \
            .fit(self.x_train, self.y_train)  # .ravel())

        self.grids[name] = grid

        results = {
            'Best params': grid.best_params_,
            'Best score': f'{grid.best_score_:.3f}'}

        f.pretty_dict(results)

        return grid

    def save_model(self, name: str, **kw) -> None:
        model = self.get_model(name=name, **kw)
        p = cf.p_res / 'models'
        f.save_pickle(model, p, name)

    def load_model(self, name: str, **kw) -> Any:
        p = p = cf.p_res / f'models/{name}.pkl'
        return f.load_pickle(p)

    def make_train_test(
            self,
            df: pd.DataFrame,
            target: List[str] = None,
            split_date: dt = None,
            train_size: float = 0.8,
            **kw):
        """Make x_train, y_train etc from df

        Parameters
        ---------
        target : list
            target column to remove for y_
        """
        if target is None:
            target = self.target

        if split_date is None:
            if not 'test_size' in kw:
                kw['train_size'] = train_size

            df_train, df_test = train_test_split(
                df, random_state=self.random_state, **kw)
        else:
            df_train = df[df.index < split_date]
            df_test = df[df.index >= split_date]

        x_train, y_train = split(df_train, target=target)
        x_test, y_test = split(df_test, target=target)

        f.set_self(vars(), exclude=('target',))

        return x_train, y_train, x_test, y_test

    def shap_plot(self, name: str, n_sample: int = 10_000, **kw):
        """Convenience wrapper for shap_plot from ModelManager"""
        sm = ShapManager(
            x=self.x_train,
            y=self.y_train,
            ct=self.ct,
            model=self.models[name],
            n_sample=n_sample)

        sm.plot(plot_type='violin', **kw)


class ShapManager():
    """Helper to manage shap values and plots"""

    def __init__(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            ct: ColumnTransformer,
            model: BaseEstimator,
            n_sample: int = 10_000,
            weights: pd.Series = None):
        """
        Parameters
        ----------
        x : pd.DataFrame
            x_train/test
        y : pd.Series
            y_train/test
        ct : ColumnTransformer
        model : BaseEstimator
        n_sample : int, optional
            by default 2000
        weights: pd.Series, optional
            pass in weights so don't have to fit every time, default None
        """
        bs = BlobStorage(container=cf.p_data / 'feats')
        f.set_self(vars())
        self._shap_values = None

    @property
    def shap_values(self) -> List[np.ndarray]:
        """List of 2 (if two classes eg -1/1) arrays of shape (self.n_sample, self.x_enc.columns)
        - eg [(10_000, 950), ...]
        """
        return self._shap_values

    def check_init(self):
        """Check if shap explainer init"""
        if not hasattr(self, 'explainer'):
            self.init_explainer()

    def init_explainer(self) -> None:
        """Create shap values/explainer to be used with summary or force plot, set to self
        - This can take ~20s to fit (cause fitting all 1000 columns)
        - NOTE OHLC cols ARE needed to get sample_weight... could pass in weights to just merge instead

        Sets
        ----
        Tuple[shap.TreeExplainer, List[np.array], pd.DataFrame, pd.DataFrame]
        """
        weights = self.weights
        if weights is None:
            weights = sg.WeightedPercentMaxMin(8, weight_linear=True).get_weight(self.x)

        weights = weights.loc[self.x.index]

        if not self.ct is None:
            # ensure cols are dropped/transformed correctly
            data = self.ct.fit_transform(self.x)
            x_enc = df_transformed(data=data, ct=self.ct)
        else:
            # just drop drop_cols
            # NOTE this doesn't normalize anything
            x_enc = self.x.pipe(f.safe_drop, cf.config['drop_cols'])

        self.model.fit(x_enc, self.y, **weighted_fit(name=None, weights=weights))

        # use smaller sample to speed up plot
        # NOTE sample will change every time new rows are added due to random_state
        x_sample = x_enc.sample(self.n_sample, random_state=0)

        self.explainer = shap.TreeExplainer(self.model)
        self._shap_values = self.explainer.shap_values(x_sample)
        self.x_sample = x_sample
        self.x_enc = x_enc

    def plot(self, plot_type: str = 'violin') -> None:
        """Show shap summary plot"""
        self.check_init()

        # bar plot needs both classes
        vals = self.shap_values
        if plot_type == 'violin':
            vals = vals[0]

        shap.summary_plot(
            shap_values=vals,
            features=self.x_sample,
            plot_type=plot_type,
            axis_color='white')

    def force_plot(self, sample_n: int = 0) -> None:
        """Show force plot for single sample

        Parameters
        ----------
        sample_n : int, optional
            row to show from self.samples, default 0
        """
        self.check_init()
        shap.initjs()
        shap.force_plot(
            self.explainer.expected_value[1],
            self.shap_values[0][sample_n, :],
            self.x_enc.iloc[sample_n, :])

    def shap_n_important(
            self,
            n: int = 50,
            as_list: bool = True,
            save: bool = True,
            save_least_all: bool = False,
            upload: bool = False) -> Union[list, pd.DataFrame]:
        """Get list of n most important shap values

        Parameters
        ----------
        n : int, optional
            num important feats, default 50
        as_list : bool, optional
            return as list or DataFrame, default True
        save : bool, optional
            save important feats to pickle, default True
        upload : bool, optional
            upload least_imp_cols to azureblob, default False

        Returns
        -------
        Union[list, pd.DataFrame]
        """
        self.check_init()

        # get correct column transformer index for numeric
        if not self.ct is None:
            transformer = 'numeric'
            transformers = self.ct.transformers
            t_idx = [t[0] for t in transformers].index(transformer)
            col_names = transformers[t_idx][2]  # 2 is column names
        else:
            col_names = self.x_enc.columns.tolist()

        # 0/1 could be positive/negative class, not sure which
        # NOTE could maybe try mean of both?
        data = np.abs(self.shap_values[0]).mean(axis=0)
        # data2 = np.abs(self.shap_values[1]).mean(axis=0)
        # data = np.vstack((data1, data2)).T

        # cols = ['cls_0', 'cls_1']
        # .assign(importance=lambda x: x[cols].mean(axis=1)) \
        # cols = ['importance']
        df = pd.DataFrame(data=data, index=col_names, columns=['importance']) \
            .sort_values('importance', ascending=False)

        m_imp = dict(
            most=df.iloc[: n].index.tolist(),
            least=df.iloc[n:].index.tolist())

        # save least important (500 important)
        if save_least_all:
            f.save_pickle(df.iloc[500:].index.tolist(), p=self.bs.p_local, name='least_imp_cols_500')

        if save:
            # for name, lst in m_imp.items():
            if not m_imp['least']:
                log.warning('No least_imp_cols to save')
            else:
                f.save_pickle(m_imp['least'], p=self.bs.p_local, name='least_imp_cols')

                if upload:
                    self.bs.upload_dir()

        return m_imp if as_list else df


def shap_top_features(shap_vals, X_sample):
    vals = np.abs(shap_vals).mean(0)
    return pd \
        .DataFrame(
            data=list(zip(X_sample.columns, vals)),
            columns=['feature_name', 'importance']) \
        .sort_values(by=['importance'], ascending=False)


def show_prop(df, target_col='target'):
    """Show proportion of classes in target column"""
    return df \
        .groupby(target_col) \
        .agg(num=(target_col, 'size')) \
        .assign(prop=lambda x: x.num / x.num.sum()) \
        .style \
        .format(dict(
            num='{:,.0f}',
            prop='{:.2%}'))


def split(df: pd.DataFrame, target: Union[list, str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Split off target col to make X and y"""
    if isinstance(target, list) and len(target) == 1:
        target = target[0]

    return df.pipe(f.safe_drop, cols=target), df[target]  # .to_numpy(np.float32)


def show_scores(df: pd.DataFrame, higher_better: bool = False) -> None:
    from jambot.utils.styles import bg, get_style
    subset = [col for col in df.columns if any(
        item in col for item in ('test', 'train')) and not 'std' in col]

    style = get_style(df) \
        .pipe(bg, higher_better=higher_better) \
        .pipe(bg, subset=subset)

    display(style)


def append_mean_std_score(
        df: pd.DataFrame = None,
        scores: dict = None,
        name: str = None,
        show: bool = False,
        scoring: dict = None) -> pd.DataFrame:
    """Create df with mean and std of all scoring metrics

    Parameters
    ----------
    df : pd.DataFrame, optional
        df of scores, by default None
    scores : dict, optional
        results from cross_val scoring, by default None
    name : str, optional
        model name, by default None
    show : bool, optional
        display scores df, by default False
    scoring : dict, optional
        scoring funcs, by default None

    Returns
    -------
    pd.DataFrame
        df with mean/std of scoring dict
    """
    if df is None:
        df = pd.DataFrame()

    # assume preprocessor then model name in pipeline.steps[1]
    if isinstance(name, Pipeline):
        name = name.steps[1][0]

    # NOTE specific to strat scorer
    exclude = ['train_max', 'train_final']
    exclude_std = ['fit_time', 'score_time'] + exclude

    def name_cols(cols, type_):
        return {col: f'{type_}{col}' for col in cols}

    score_cols = [col for col in scores.keys() if not col in exclude_std]
    mean_cols = name_cols(score_cols, '')
    std_cols = name_cols(score_cols, 'std_')

    # this is a series for scores from a single cv run
    s_scores = pd.DataFrame(scores) \
        .mean() \
        .rename(mean_cols) \
        .append(
            pd.DataFrame(scores)
            .pipe(f.safe_drop, exclude_std)
            .std()
            .rename(std_cols)) \
        .drop(exclude)  # only drop for non regression

    df.loc[name, s_scores.index] = s_scores

    # flip sign of mean cols for scorer where lower is better, eg MASE
    if scoring:
        for scorer_name, scorer in scoring.items():

            if (hasattr(scorer, '_sign') and scorer._sign == -1) or 'neg' in str(scorer):
                scorer_cols = [c for c in df.columns if not 'std' in c and all(
                    item in c.split('_') for item in (scorer_name,))]

                df.loc[name, scorer_cols] = df.loc[name, scorer_cols] * -1

    if show:
        from jambot.utils.styles import get_style
        display(get_style(df))

    return df


def df_transformed(data, ct):
    """Return dataframe of transformed df with correct feature names

    Parameters
    ----------
    data : np.ndarray
        transformed data from ColumnTransformer
    ct : ColumnTransFormer

    Returns
    -------
    pd.DatFrame
    """
    cols = get_ct_feature_names(ct=ct)
    cols = [col[1] for col in cols]  # convert tuples back to list
    # print(cols)
    return pd.DataFrame(data, columns=cols)


def df_coef(ct, model, num_features=20, best=True, round_coef=3, feat_imp=False) -> pd.DataFrame:
    """Get df of feature names with corresponding coefficients

    Parameters
    ----------
    ct : ColumnTransformer
    model : any
        sklearn model which implements `.coef_`
    num_features : int
        number of top features ranked by coefficient. Pass -1 to get all features
    best : bool
        if true, return best features (positive coef), else return worst features (negative coef)
    round_coef : int
        round coef column, default 3,
    feat_imp : bool, default False
        use 'feature_importances_' instead of coef

    Returns
    -------
    pd.DataFrame
        DataFrame of transformer name, feature name, coefficient
    """
    coef = model.coef_[0] if not feat_imp else model.feature_importances_

    m = dict(
        # this is a tuple of (transformer, feature_name)
        transformer_feature=get_ct_feature_names(ct),
        coef=coef)

    return pd.DataFrame(m) \
        .pipe(lambda df: pd.concat([
            df,
            pd.DataFrame(df['transformer_feature'].to_list())], axis=1)) \
        .rename(columns={0: 'transformer', 1: 'feature'}) \
        .drop(columns=['transformer_feature']) \
        .sort_values('coef', ascending=not best)[:num_features] \
        .assign(coef=lambda x: x.coef.round(round_coef))[['transformer', 'feature', 'coef']]


def get_feature_out(estimator, feature_in):

    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return estimator.get_feature_names()  # don't prepend 'vec'
            # return [f'vec_{f}' \
            #     for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    """
    Code adapted from https://stackoverflow.com/a/57534118/6278428
    - handles all estimators, pipelines inside ColumnTransfomer
    - doesn't work when remainder =='passthrough', which requires the input column names.
    """

    output_features = []

    # keep name associate with feature for further filtering in df
    make_tuple = lambda name, features: [(name, feature_name) for feature_name in features]

    for name, estimator, features in ct.transformers_:
        if estimator == 'drop':
            pass
        elif not name == 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(make_tuple(name, features_out))
        elif estimator == 'passthrough':
            output_features.extend(make_tuple(
                name, ct._feature_names_in[features]))

    # print(output_features)
    return output_features


def mpl_dict(params):
    """"Convert _ to . for easier mpl rcparams grid definition"""
    return {k.replace('_', '.'): v for k, v in params.items()}


def all_except(df, exclude: list):
    """Return all cols in df except exclude

    Parameters
    ----------
    df : pd.DataFrame
    exclude : list | iterable
        column names to exclude

    Returns
    -------
    list
        list of all cols in df except exclude
    """
    return [col for col in df.columns if not any(col in lst for lst in exclude)]


def df_proba(x: pd.DataFrame, model: BaseEstimator, **kw) -> pd.DataFrame:
    """Return df of predict_proba, with timestamp index

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : BaseEstimator
        fitted model/pipeline

    Returns
    -------
    pd.DataFrame
        df with proba_ added
    """
    arr = model.predict_proba(x)
    m = {-1: 'short', 0: 'neutral', 1: 'long'}
    cols = [f'proba_{m.get(c)}' for c in model.classes_]
    return pd.DataFrame(data=arr, columns=cols, index=x.index)


def df_y_pred(x: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """Return df with y_pred added

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : BaseEstimator
        model/pipe

    Returns
    -------
    pd.DataFrame
        df with y_pred added
    """
    return pd.DataFrame(
        data=model.predict(x),
        columns=['y_pred'],
        index=x.index)


def add_probas(df: pd.DataFrame, model: BaseEstimator, x: pd.DataFrame, do: bool = True, **kw) -> pd.DataFrame:
    """Convenience func to add probas if don't exist

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : BaseEstimator
        model/pipe
    x : pd.DataFrame
        section of df to predict on
    do : bool, optional
        pass False to skip (for regression)

    Returns
    -------
    pd.DataFrame
        df with proba_ added
    """

    # already added probas
    if not do or 'proba_long' in df.columns:
        return df

    return df.pipe(f.left_merge, df_right=df_proba(x=x, model=model))


def add_y_pred(df: pd.DataFrame, model: BaseEstimator, x: pd.DataFrame) -> pd.DataFrame:
    """Add y_pred col to df with model.predict

    Parameters
    ----------
    df : pd.DataFrame
        df with signals
    model : BaseEstimator
        model/pipe
    x : pd.DataFrame
        section of df to predict on

    Returns
    -------
    pd.DataFrame
        df with y_pred added
    """
    return df.pipe(f.left_merge, df_right=df_y_pred(x=x, model=model))


def convert_proba_signal(
        df: pd.DataFrame,
        col: str = 'rolling_proba',
        regression: bool = False) -> pd.DataFrame:
    """Convert probas btwn 0-1 to a signal of 0, 1 or -1 with threshold 0.5

    Parameters
    ----------
    df : pd.DataFrame
        df with rolling proba col
    col : str
        col to conver to signal, default 'rolling_proba'

    Returns
    -------
    pd.DataFrame
        df with signal added
    """
    s = df[col]

    # for regression, just using y_pred which is already pos/neg, not proba btwn 0-1
    offset = 0.5 if not regression else 0

    return df \
        .assign(
            signal=np.sign(np.diff(np.sign(s - offset), prepend=np.array([0]))))


def weighted_score(
        y_true: pd.Series,
        y_pred: np.ndarray,
        weights: pd.Series) -> float:
    """Match weights with y_true series to get a weighted accuracy

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    weights : pd.Series
        from sg.WeightedScorer

    Returns
    -------
    float
    """

    # slice full weights
    weights = weights.loc[y_true.index]
    return np.mean(np.where(y_true.astype(int) == y_pred.astype(int), 1, -1) * weights) * 100


def smape(y_true, y_pred, h=1, **kw):
    """Calculate symmetric mean absolute percentage error

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    h : int, optional
        The forecast horizon, by default 1

    Returns
    -------
    float :
        The sMAPE of the `y_pred` against `y_true`
    """
    return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) * h))


def mase(y_true, y_pred, h=1, **kw):
    """Calculates the mean averaged scaled error for a time series by comparing
    to the naive forecast (shift h prior value forward as the forecast)
    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    h : int, optional
        The forecast horizon, by default 1
    Returns
    -------
    float :
        The MASE for `y_true` and `y_pred`
    """
    d = np.abs(np.diff(y_true)).sum() / (y_pred.shape[0] - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / (d * h)


def avg_mase_smape(y_true, y_pred, h=1, **kw):
    """Calculates the average of MASE and SMAPE for time series predictions
    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    h : int, optional
        The forecast horizon, by default 1
    Returns
    -------
    float :
        The (SMAPE + MASE)/2 for `y_true` and `y_pred`
    """
    return (smape(y_true, y_pred, h=h) + mase(y_true, y_pred, h=h)) / 2


def reverse_pct(df, start_num, pct_col, num_col):
    """Convert % change back to number"""
    nums_out = []
    for i in range(df.shape[0]):
        start_num = df[pct_col].iloc[i] * start_num + start_num
        nums_out.append(start_num)

    return df \
        .assign(**{num_col: nums_out})


def add_preds_minmax(df, features: dict, encoders: dict, train_size: float) -> pd.DataFrame:
    """Add pred_max, pred_min"""
    features = copy.copy(features)
    multi_target = ['target_max', 'target_min']
    pred_cols = ['pred_max', 'pred_min']
    features['target'] = multi_target
    features['drop'] = [c for c in features['drop'] if not any(
        c in item for item in [multi_target, pred_cols])] + ['target']

    sm = sg.SignalManager(slope=5)

    signals = [
        sg.TargetMaxMin(n_periods=4),
        # 'Volatility',
        # 'Volume'
    ]

    df = df \
        .pipe(sm.add_signals, signals=signals) \
        .pipe(lambda df: df.fillna(df.mean()))

    mm2 = ModelManager(
        features=features,
        encoders=encoders)

    # model_multi = MultiOutputRegressor(
    #     LGBMRegressor(num_leaves=50, n_estimators=50, max_depth=-1, boosting_type='goss'))

    model_multi = MultiOutputRegressor(Ridge())

    steps = [
        (1, ('pca', PCA(n_components=20, random_state=0)))]
    # steps = None

    pipe_multi = mm2.make_pipe(
        name='lgbm_maxmin',
        model=model_multi,
        steps=steps)

    x_train, y_train, x_test, _ = mm2.make_train_test(
        df=df,
        target=features['target'],
        train_size=train_size,
        shuffle=False)

    pipe_multi.fit(x_train, y_train)

    df_pred_multi = pd.DataFrame(
        data=pipe_multi.predict(x_test),
        columns=pred_cols,
        index=x_test.index)

    # NOTE this might be some data leakage... idk probs not too severe
    return df.pipe(f.left_merge, df_pred_multi) \
        .fillna(0)
    # .pipe(lambda df: df.fillna(df.mean()))


def as_multi_out(models: dict) -> Dict[str, MultiOutputRegressor]:
    """Convert dict of models to MultiOutputRegressors for predicting more than one target"""
    return {k: MultiOutputRegressor(model) for k, model in models.items()}


def weighted_fit(name: str = None, n: int = None, weights: np.ndarray = None) -> Dict[str, np.ndarray]:
    """Create dict of weighted samples for fit params

    Parameters
    ----------
    name : str, optional
        model name to prepend to dict key, by default None
    n : int, optional
        length of array, by default None
    weights : np.ndarray, optional
        allow passing in other weights to just use this for dict formatting, by default None

    Returns
    -------
    Dict[str, np.ndarray]
        {lgbm__sample_weight: [0.5, ..., 1.0]}
    """
    name = f'{name}__' if not name is None else ''

    if weights is None:
        weights = np.linspace(0.5, 1, n)

    return {f'{name}sample_weight': weights}


def plot_pred_dist(df: pd.DataFrame, cols: list = None) -> None:
    """Show density plots of distribution target vs y_pred classes"""
    if cols is None:
        cols = ('target', 'y_pred')

    fig, axs = plt.subplots(nrows=len(cols), sharex=True, figsize=(8, 10))

    for col, ax in zip(cols, axs):
        df[col].value_counts().sort_values() \
            .plot(kind='bar', ax=ax, title=col)


def plot_cols(df: pd.DataFrame, expr: str = '.', cols: List[str] = None) -> None:
    """Plot all cols filtered by regex expr"""
    if cols is None:
        cols = [c for c in df.columns if re.search(expr, c)]

    ncols = len(cols)

    fig, axs = plt.subplots(nrows=ncols, sharex=True, figsize=(12, 2 * ncols))

    for col, ax in zip(cols, axs):
        df[col].plot(title=col, ax=ax)


def is_fitted(estimator: BaseEstimator) -> bool:
    """Check if model is fitted yet

    Parameters
    ----------
    estimator : BaseEstimator

    Returns
    -------
    bool
        if fitted
    """
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False
