import copy
import re
import time
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from jambot import config as cf
from jambot import display, getlog
from jambot import signals as sg
from jambot.common import ProgressParallel
from jambot.ml import models as md
from jambot.weights import WeightsManager
from jgutils import fileops as jfl
from jgutils import functions as jf
from jgutils import pandas_utils as pu
from jgutils.azureblob import BlobStorage

# not needed when running on azure
try:
    import shap
    from icecream import ic

    # from tqdm import tqdm
    ic.configureOutput(prefix='')

    import matplotlib.pyplot as plt

except (ImportError, ModuleNotFoundError) as e:
    pass

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

log = getlog(__name__)


class ModelManager(object):
    """Manager class to perform cross val etc on multiple models with same underlying column transformer + data
    """
    ct: ColumnTransformer

    def __init__(
            self,
            ct: ColumnTransformer = None,
            scoring=None,
            cv_args: Optional[dict] = None,
            random_state: int = 0,
            target: str = 'target',
            scorer=None,
            wm: WeightsManager = None,
            **kw):
        self.cv_args = cv_args if not cv_args is None else {}
        self.df_results = pd.DataFrame()
        pipes = {}
        scores = {}
        models = {}
        grids = {}
        df_preds = {}
        cv_data = {}  # used to return estimator/strats for each cv fold
        jf.set_self(include=kw)

        self.wm = wm

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
        jf.set_self()
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

        jf.set_self()
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
            for step in jf.as_list(steps):
                pipe.steps.insert(step[0], step[1])

        self.pipes[name] = pipe

        return pipe

    def init_models(self, models: Dict[str, 'BaseEstimator'], **kw) -> None:
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
            model: 'BaseEstimator',
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
            extra_cv_args: dict = None,
            filter_train_quantile: float = None):
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

            # if not filter_train_quantile is None:
            #     # TODO not done
            #     x_train, y_train = self.wm.filter_quantile(
            #         datas=[self.x_train, self.y_train],
            #         quantile=filter_train_quantile)
            # else:
            x_train, y_train = self.x_train, self.y_train

            scores = cross_validate(pipe, x_train, y_train, error_score='raise', **self.cv_args)

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
            model: 'BaseEstimator',
            weighted_fit: bool = True,
            filter_fit_quantile: float = None,
            name: str = None,
            # fit_params: dict = None,
            force: bool = False,
    ) -> 'BaseEstimator':
        """Fit model to self.x_train with self.y_train (if not fit already)

        Parameters
        ----------
        model : BaseEstimator
            estimator/pipeline
        fit_params : dict, optional
            eg lgbm__sample_weight, by default None
        force: : bool, optional
            sometimes just need to pipe this but not fit again
        filter_fit_quantile : float, optional
            filter data before fitting to highest quantile

        Returns
        -------
        BaseEstimator
            fitted model
        """
        x_train, y_train = self.x_train, self.y_train

        if not filter_fit_quantile is None:
            x_train, y_train = self.wm.filter_quantile(
                datas=[x_train, y_train],
                quantile=filter_fit_quantile)

        fit_params = self.wm.fit_params(x_train, name=name) if weighted_fit else {}

        if force or not is_fitted(model):
            log.info(f'Fitting model, x_train: {x_train.shape[0]}')
            model.fit(x_train, y_train, **fit_params)
        else:
            log.warning('Not fitting model')

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

    def add_predict(
            self,
            df: pd.DataFrame,
            weighted_fit: bool = True,
            filter_fit_quantile: float = None,
            x_train: pd.DataFrame = None,
            x_test: pd.DataFrame = None,
            proba: bool = True,
            model: 'BaseEstimator' = None,
            name: str = None,
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

        model = self.fit(
            name=name,
            model=model,
            force=True,
            weighted_fit=weighted_fit,
            filter_fit_quantile=filter_fit_quantile)

        if x_test is None:
            x_test = self.x_test

        df = df \
            .pipe(md.add_y_pred, model=model, x=x_test) \
            .pipe(md.add_probas, model=model, x=x_test, do=proba)

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
                param_name='param_grid'))[search_type]

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

        jf.pretty_dict(results)

        return grid

    def save_model(self, name: str, **kw) -> None:
        model = self.get_model(name=name, **kw)
        p = cf.p_res / 'models'
        jfl.save_pickle(model, p, name)

    def load_model(self, name: str, **kw) -> Any:
        p = p = cf.p_res / f'models/{name}.pkl'
        return jfl.load_pickle(p)

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

        x_train, y_train = pu.split(df_train, target=target)
        x_test, y_test = pu.split(df_test, target=target)

        jf.set_self(exclude=('target',))

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
            df: pd.DataFrame,
            model: 'BaseEstimator',
            n_sample: int = 10_000,
            wm: WeightsManager = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            full df, x_train/test + y
        model : BaseEstimator
        n_sample : int, optional
            by default 2000
        wm: WeightsManager, optional
            initialized WeightsManager
        """
        if wm is None:
            wm = WeightsManager.from_config(df)

        self.wm = wm
        self.bs = BlobStorage(container=cf.p_data / 'feats')
        self.df = df
        self.model = model
        self.n_sample = n_sample
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
        df = self.df
        df = self.wm.filter_quantile(df, quantile=cf.FILTER_FIT_QUANTILE)

        x = df.drop(columns=['target'])
        y = df.target

        # NOTE this doesn't normalize anything
        x_enc = x.pipe(pu.safe_drop, cf.DROP_COLS)

        self.model.fit(x_enc, y, **self.wm.fit_params(x=x_enc, name=None))

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
            upload: bool = False) -> Union[Dict[str, List[str]], pd.DataFrame]:
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
        # if not self.ct is None:
        #     transformer = 'numeric'
        #     transformers = self.ct.transformers
        #     t_idx = [t[0] for t in transformers].index(transformer)
        #     col_names = transformers[t_idx][2]  # 2 is column names
        # else:
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
            jfl.save_pickle(df.iloc[500:].index.tolist(), p=self.bs.p_local, name='least_imp_cols_500')

        if save:
            # for name, lst in m_imp.items():
            if not m_imp['least']:
                log.warning('No least_imp_cols to save')
            else:
                p = jfl.save_pickle(m_imp['least'], p=self.bs.p_local, name='least_imp_cols')
                log.info(f'saved: {p}')

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
            .pipe(pu.safe_drop, exclude_std)
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


def weighted_score(
        y_true: pd.Series,
        y_pred: pd.Series,
        weights: pd.Series) -> float:
    """Match weights with y_true series to get a weighted accuracy

    Parameters
    ----------
    y_true : pd.Series
    y_pred : pd.Series
    weights : pd.Series
        from sg.WeightedScorer

    Returns
    -------
    float
    """
    if not isinstance(y_pred, pd.Series):
        log.warning('converting y_pred to pd.Series')
        y_pred = pd.Series(y_pred, index=y_true.index)

    # drop NA from y_pred, match y_true index
    y_pred = y_pred.dropna()
    y_true = y_true.loc[y_pred.index]

    # slice full weights
    weights = weights.loc[y_true.index]
    return np.mean(np.where(y_true.astype(int) == y_pred.astype(int), 1, -1) * weights) * 100


def accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Simple wrapper around sklearn accuracy_score to drop NAs"""
    y_pred = y_pred.dropna()
    y_true = y_true.loc[y_pred.index]
    return sk_accuracy_score(y_true, y_pred)


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
    return df.pipe(pu.left_merge, df_pred_multi) \
        .fillna(0)
    # .pipe(lambda df: df.fillna(df.mean()))


def as_multi_out(models: dict) -> Dict[str, MultiOutputRegressor]:
    """Convert dict of models to MultiOutputRegressors for predicting more than one target"""
    return {k: MultiOutputRegressor(model) for k, model in models.items()}


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


def is_fitted(estimator: 'BaseEstimator') -> bool:
    """Check if model is fitted yet

    Parameters
    ----------
    estimator : 'BaseEstimator'

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


def _get_imp_feats(
        x: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame) -> List[str]:
    """Fit shap on history up to this point
    - NOTE not used currently
    - use n most important cols going forward
    - NOTE could limit fitting important cols up to eg last year?
    """
    # need to pass in x/y for history till now
    # need to know if we need ct for column names or not
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        num_leaves=50,
        n_estimators=50,
        max_depth=30,
        boosting_type='dart',
        learning_rate=0.1)

    spm = ShapManager(
        df=df,
        model=model,
        n_sample=10_000)

    return spm.shap_n_important(n=60, save=False, upload=False, as_list=True)['most']


def add_predict_iter(
        df: pd.DataFrame,
        wm: WeightsManager,
        model: 'BaseEstimator',
        split_date: dt,
        batch_size: int = 96,
        max_train_size: int = None,
        regression: bool = False,
        filter_fit_quantile: float = None,
        n_jobs: int = -1,
        retrain_feats: bool = False,
        interval: int = 15) -> pd.DataFrame:
    """Retrain model every x periods and add predictions for next batch_size

    Parameters
    ----------
    df : pd.DataFrame
    wm : WeightsManager
    model : BaseEstimator
    split_date : dt
        first date to start predicting
    batch_size : int, optional
        size in periods, by default 96
    max_train_size : int, optional
        size in months, by default None
    regression : bool, optional
        default False
    filter_fit_quantile : float, optional
        default None
    n_jobs : int, optional
    retrain_feats : bool, optional
        retrain important feats (not used), by default False
    interval : int, optional
        period interval, by default 15

    Returns
    -------
    pd.DataFrame
        df with predictions added
    """
    df_orig = df.copy()  # to keep all original cols when returned
    delta_mins = {1: 60, 15: 15}[interval]

    # nrows = df.shape[0]
    # nrows = df.groupby('symbol').size().max()  # TODO not quite, need diff of higest - lowest date
    # index = df.index  # type: pd.MultiIndex #(str, pd.DatetimeIndex)
    dt_index = df.index.get_level_values('timestamp')  # type: pd.DatetimeIndex
    d_start = dt_index.min().to_pydatetime()
    n_periods = (dt_index.max() - split_date).to_pytimedelta().total_seconds() / 60 / delta_mins

    # min_size = df[index < split_date].shape[0]  # type: int
    # num_batches = ((nrows - min_size) // batch_size) + 1
    num_batches = int((n_periods // batch_size) + 1)

    # if model is None:
    #     model = self.pipes[name].named_steps[name]

    # convert months to periods  NOTE not implemented for date indexing yet
    # if not max_train_size is None:
    #     max_train_size = max_train_size * 30 * 24 * {1: 1, 15: 4}[interval]

    df = df.pipe(pu.safe_drop, cf.DROP_COLS)

    def _fit(i: int) -> Union[pd.DataFrame, None]:

        # i_lower = min_size + i * batch_size  # "test" lower
        d_lower = split_date + delta(minutes=i * batch_size * delta_mins)
        d_upper = d_lower + delta(minutes=batch_size * delta_mins)
        # i_upper = min(i_lower + batch_size, nrows + 1)
        # idx_test = df.index[i_lower: i_upper]
        idx_test_slice = pd.IndexSlice[:, d_lower: d_upper + delta(minutes=-delta_mins)]

        # train model from 0 up to current position
        # NOTE max_train_size eg 2 yrs seems to be much worse
        # i_train_lower = 0 if not max_train_size else max(0, i_lower - max_train_size)

        # df_train = df.iloc[i_train_lower: i_lower]  # type: pd.DataFrame
        df_train = df.loc[pd.IndexSlice[:, d_start: d_lower + delta(minutes=-delta_mins)], :]

        if filter_fit_quantile:
            # TODO possibly filter highest PER SYMBOL, not all combined
            # NOTE this is cheating, wm has knowledge of future values
            df_train = wm.filter_quantile(df_train, quantile=filter_fit_quantile, _log=False)

        x_train, y_train = pu.split(df_train)
        x_test, _ = pu.split(df.loc[idx_test_slice, :])

        model.fit(
            x_train,
            y_train,
            **wm.fit_params(x_train))

        if len(idx_test_slice) == 0:
            return None
        else:
            if not regression:
                proba_long = md.df_proba(x=x_test, model=model)['proba_long']
            else:
                proba_long = md.df_y_pred(x=x_test, model=model)['y_pred']

            return pd.DataFrame(
                # index=x_test.loc[idx_test_slice, :].index,  # don't need explicit index, uses index from df_proba
                data=dict(
                    y_pred=model.predict(x_test),
                    proba_long=proba_long))

    par = ProgressParallel(batch_size=2, n_jobs=n_jobs, total=num_batches)
    result = par(delayed(_fit)(i=i) for i in range(num_batches))

    # result = [_fit(i) for i in range(num_batches)]

    return df_orig.pipe(pu.left_merge, pd.concat([df for df in result if not df is None]))
