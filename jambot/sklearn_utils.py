import copy
import json
import pickle
import re
import time
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from icecream import ic
from IPython.core.display import display_pdf
from IPython.display import display
from lightgbm import LGBMRegressor
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
from matplotlib.colors import ListedColormap, TwoSlopeNorm, rgb2hex, to_hex
from seaborn import diverging_palette
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             make_scorer, mean_squared_error, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from jambot import functions as f
from jambot import signals as sg

ic.configureOutput(prefix='')
try:
    import shap

except ImportError as e:
    # print('failed to import module', e)
    pass

# p_tmp = Path.home() / 'Desktop/sklearn_temp'
# print(p_tmp.exists())
_cmap = diverging_palette(240, 10, sep=10, n=21, as_cmap=True)


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
            **kw):
        cv_args = cv_args if not cv_args is None else {}
        df_results = pd.DataFrame()
        pipes = {}
        scores = {}
        models = {}
        grids = {}
        df_preds = {}
        v = {**vars(), **kw}
        set_self(v)

        if any(item in kw for item in ('features', 'encoders')):
            self.make_column_transformer(**kw)

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

        set_self(vars())
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
        print(df_trans.shape)
        display(df_trans.describe().T)

    def get_model(self, name: str, best_est=False):
        if best_est:
            return self.best_est(name=name)
        else:
            return self.models[name]

    def cross_val_feature_params(self, signal_manager, name, model, feature_params: dict, train_size: float = 0.8):
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

    def make_pipe(self, name: str, model, steps: list = None):
        pipe = Pipeline(
            steps=[
                ('ct', self.ct),
                (name, model)],
            # ('pca', PCA(n_components=10, random_state=self.random_state)),
        )

        # insert extra steps in pipe, eg RFECV
        if not steps is None:
            if not isinstance(steps, list):
                steps = [steps]
            for step in steps:
                pipe.steps.insert(step[0], step[1])

        self.pipes[name] = pipe

        return pipe

    def cross_val(self, models: dict, show: bool = True, steps: list = None, df_scores=None):
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

        for name, model in models.items():

            # allow passing model definition, or instantiated model
            if isinstance(model, type):
                model = model()

            model.random_state = self.random_state

            # safe model/pipeline by name
            self.models[name] = model

            pipe = self.make_pipe(name=name, model=model, steps=steps)

            scores = cross_validate(
                pipe, self.x_train, self.y_train, error_score='raise', **self.cv_args)  # .ravel()

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

    def fit(self, name: str, best_est=False, model=None, fit_params=None):
        """Fit model to training data"""
        if best_est:
            model = self.best_est(name)

        if model is None:
            model = self.pipes[name]

        # allow passing in fit params eg lgbm__sample_weight=..
        fit_params = fit_params or {}

        model.fit(self.x_train, self.y_train, **fit_params)  # .ravel()
        return model

    def y_pred(self, X, model=None, **kw):
        if model is None:
            model = self.fit(**kw)

        return model.predict(X)

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

    def df_proba(self, df=None, model=None, **kw):
        """Return df of predict_proba, with timestamp index"""
        if df is None:
            df = self.x_test

        if model is None:
            model = self.fit(**kw)

        arr = model.predict_proba(df)
        m = {-1: 'short', 0: 'neutral', 1: 'long'}
        cols = [f'proba_{m.get(c)}' for c in model.classes_]
        return pd.DataFrame(data=arr, columns=cols, index=df.index)

    def add_proba(self, df, do=False, **kw):
        """Concat df of predict_proba"""
        return pd.concat([df, self.df_proba(**kw)], axis=1) if do else df

    def add_predict_iter(
            self,
            df,
            name: str,
            model=None,
            pipe=None,
            batch_size: int = 96,
            min_size: int = 180 * 24,
            max_train_size=None,
            regression=True):
        """Retrain model every x periods and add predictions for next batch_size"""
        df_train = df.copy()
        df = df.copy()

        df['y_pred'] = np.NaN
        if not regression:
            df['proba_long'] = np.NaN

        nrows = df.shape[0]
        num_batches = ((nrows - min_size) // batch_size) + 1

        if pipe is None:
            pipe = self.make_pipe(name=name, model=model)

        # return num_batches
        for i in tqdm(range(num_batches)):

            i_lower = min_size + i * batch_size
            i_upper = min(i_lower + batch_size, nrows + 1)
            idx = df.index[i_lower: i_upper]

            # max number of rows to train on
            # if max_train_size is None:
            #     max_train_size = i_lower

            # train model up to current position
            x_train, y_train = split(
                df_train.iloc[0: i_lower],
                target=self.target)

            pipe.fit(
                x_train,
                y_train,
                lgbm__sample_weight=np.linspace(0.5, 1, x_train.shape[0]))

            # add preds to model
            x_test, _ = split(df_train.loc[idx], target=self.target)
            y_pred = pipe.predict(x_test)
            y_true = df.loc[idx, self.target]

            if not regression:
                df.loc[idx, 'proba_long'] = self.df_proba(df=x_test, model=pipe)['proba_long'].values

            # rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)

            df.loc[idx, 'y_pred'] = y_pred

        df_final = df[[self.target, 'y_pred']].dropna()
        # rmse_final = mean_squared_error(
        #     y_true=df_final[self.target].values,
        #     y_pred=df_final['y_pred'].values,
        #     squared=False)
        # print(f'\nrmse_final: {rmse_final:.4f}')

        return df

    def add_predict(self, df, proba=True, **kw):
        """Add predicted vals to df"""
        df = df \
            .assign(
                y_pred=self.y_pred(X=df.drop(columns=[self.target]), **kw)) \
            .pipe(self.add_proba, do=proba, **kw)

        # save predicted values for each model
        self.df_preds[kw['name']] = df

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
            print('multioutput')
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

        pretty_dict(results)

        return grid

    def save_model(self, name: str, **kw):
        model = self.get_model(name=name, **kw)

        p = f.p_res / f'/models/{name}.pkl'

        with open(p, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, name: str, **kw):
        filename = f'{name}.pkl'
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def make_train_test(self, df, target, split_date=None, train_size=0.8, **kw):
        """Make x_train, y_train etc from df

        Parameters
        ---------
        target : list
            target column to remove for y_
        """
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

        set_self(vars(), exclude=('target',))

        return x_train, y_train, x_test, y_test

    def shap_plot(self, name, **kw):
        """Convenience wrapper for shap_plot from ModelManager"""
        shap_plot(
            X=self.x_train,
            y=self.y_train,  # .ravel(),
            ct=self.ct,
            model=self.models[name],
            **kw)


def shap_explainer_values(X, y, ct, model, n_sample=2000):
    """Create shap values/explainer to be used with summary or force plot"""
    data = ct.fit_transform(X)
    X_enc = df_transformed(data=data, ct=ct)
    model.fit(X_enc, y)

    # use smaller sample to speed up plot
    X_sample = X_enc
    if not n_sample is None:
        X_sample = X_enc.sample(n_sample, random_state=0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample, X_enc


def shap_plot(X, y, ct, model, n_sample=2000):
    """Show shap summary plot"""
    explainer, shap_values, X_sample, X_enc = shap_explainer_values(X=X, y=y, ct=ct, model=model, n_sample=n_sample)

    shap.summary_plot(
        shap_values=shap_values[0],
        features=X_sample,
        plot_type='violin',
        axis_color='white')


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


def split(df, target):
    """Split off target col to make X and y"""
    if isinstance(target, list) and len(target) == 1:
        target = target[0]

    return df.drop(columns=f.as_list(target)), df[target].values


def format_cell(bg, t='black'):
    return f'background-color: {bg};color: {t};'


def bg(style, subset=None, higher_better=True, axis=0):
    """Show style with highlights per column"""
    if subset is None:
        subset = style.data.columns

    cmap = _cmap.reversed() if higher_better else _cmap

    return style \
        .background_gradient(cmap=cmap, subset=subset, axis=axis)


def background_grad_center(s, center=0, vmin=None, vmax=None, higher_better=True):
    """Style column with diverging color palette centered at value, including dark/light text formatting
    - modified from https://github.com/pandas-dev/pandas\
    /blob/b7cb3dc25a5439995d2915171c7d5172836c634e/pandas/io/formats/style.py

    Parameters
    ----------
    s : pd.Series
        Column to style
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        default self.cmap
    center : int, optional
        value to center diverging color, default 0
    vmin : float, optional
        min value, by default None
    vmax : float, optional
        max value, by default None

    Returns
    -------
    list
        list of background colors for styler
    """
    cmap = _cmap.reversed() if higher_better else _cmap

    vmin = vmin or s.values.min()
    vmax = vmax or s.values.max()

    # vmin/vmax have to be outside center
    if vmin >= center:
        vmin = center - 0.01

    if vmax <= center:
        vmax = center + 0.01

    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    text_color_threshold = 0.408  # default from pandas

    def relative_luminance(rgba) -> float:
        """Check if rgba color is greater than darkness threshold"""
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3])

        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba) -> str:
        dark = relative_luminance(rgba) < text_color_threshold
        text_color = '#f1f1f1' if dark else '#000000'
        return f'background-color: {rgb2hex(rgba)}; color: {text_color};'

    rgbas = cmap(norm(s.astype(float).values))

    if s.ndim == 1:
        return [css(rgba) for rgba in rgbas]
    else:
        return pd.DataFrame(
            [[css(rgba) for rgba in row] for row in rgbas],
            index=s.index,
            columns=s.columns)


def highlight_val(df, m: dict):

    m_replace = {k: format_cell(bg=v[0], t=v[1]) for k, v in m.items()}

    return df.replace(m_replace)


def get_style(df):
    return df.style \
        .format('{:.3f}')


def show_scores(df, higher_better=False):
    subset = [col for col in df.columns if any(
        item in col for item in ('test', 'train')) and not 'std' in col]

    style = get_style(df) \
        .pipe(bg, higher_better=higher_better) \
        .pipe(bg, subset=subset)

    display(style)


def append_fit_score(df, scores, name):

    return df


def append_mean_std_score(df=None, scores=None, name=None, show=False, scoring: dict = None):
    """Create df with mean and std of all scoring metrics"""
    if df is None:
        df = pd.DataFrame()

    if isinstance(name, Pipeline):
        # assume preprocessor then model name in pipeline.steps[1]
        name = name.steps[1][0]

    exclude = ['fit_time', 'score_time']

    def name_cols(cols, type_):
        return {col: f'{type_}{col}' for col in cols}

    score_cols = [col for col in scores.keys() if not col in exclude]
    mean_cols = name_cols(score_cols, '')
    std_cols = name_cols(score_cols, 'std_')

    df_scores = pd.DataFrame(scores).mean() \
        .rename(mean_cols) \
        .append(
            pd.DataFrame(scores)
        .drop(columns=exclude)
        .std()
        .rename(std_cols))

    df.loc[name, df_scores.index] = df_scores

    # flip sign of mean cols for scorer where lower is better, eg MASE
    if scoring:
        for scorer_name, scorer in scoring.items():

            if hasattr(scorer, '_sign') or 'neg' in str(scorer):
                scorer_cols = [c for c in df.columns if not 'std' in c and all(
                    item in c.split('_') for item in (scorer_name,))]

                df.loc[name, scorer_cols] = df.loc[name, scorer_cols] * -1

    if show:
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


def pretty_dict(m: dict, html=False, prnt=True) -> str:
    """Print pretty dict converted to newlines
    Paramaters
    ----
    m : dict
    html: bool
        Use <br> instead of html
    Returns
    -------
    str
        'Key 1: value 1
        'Key 2: value 2"
    """
    s = json.dumps(m, indent=4)
    newline_char = '\n' if not html else '<br>'

    # remove these chars from string
    remove = '}{\'"[]'
    for char in remove:
        s = s.replace(char, '')

    s = s \
        .replace(', ', newline_char) \
        .replace(',\n', newline_char)

    # remove leading and trailing newlines
    s = re.sub(r'^[\n]', '', s)
    s = re.sub(r'\s*[\n]$', '', s)

    if prnt:
        print(s)
    else:
        return s


def df_dict(m: dict, colname=None, prnt=True):
    """Quick display of dataframe from dict

    Parameters
    ----------
    m : dict
        dictionary to display
    colname : str, optional
    prnt : bool, optional
    """

    colname = colname or 'col1'
    df = pd.DataFrame.from_dict(m, orient='index', columns=[colname])

    if prnt:
        display(df)
    else:
        return df


def inverse(m: dict) -> dict:
    """Return inverse of dict"""
    return {v: k for k, v in m.items()}


def set_self(m, prnt=False, exclude=()):
    """Convenience func to assign an object's func's local vars to self"""
    if not isinstance(exclude, tuple):
        exclude = (exclude, )
    exclude += ('__class__', 'self')  # always exclude class/self
    obj = m.get('self', None)  # self must always be in vars dict

    if obj is None:
        return

    for k, v in m.items():
        if prnt:
            print(f'\n\t{k}: {v}')

        if not k in exclude:
            setattr(obj, k, v)


def remove_bad_chars(w: str):
    """Remove any bad chars " : < > | . \ / * ? in string to make safe for filepaths"""  # noqa
    return re.sub(r'[":<>|.\\\/\*\?]', '', str(w))


def to_snake(s: str):
    """Convert messy camel case to lower snake case

    Parameters
    ----------
    s : str
        string to convert to special snake case

    Examples
    --------
    """
    s = remove_bad_chars(s).strip()  # get rid of /<() etc
    s = re.sub(r'[\]\[()]', '', s)  # remove brackets/parens
    s = re.sub(r'[\n-]', '_', s)  # replace newline/dash with underscore
    s = re.sub(r'[%]', 'pct', s)
    s = re.sub(r"'", '', s)

    # split on capital letters
    expr = r'(?<!^)((?<![A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))(?=[A-Z])'

    return re \
        .sub(expr, '_', s) \
        .lower() \
        .replace(' ', '_') \
        .replace('__', '_')


def lower_cols(df):
    """Convert df columns to snake case and remove bad characters"""
    is_list = False

    if isinstance(df, pd.DataFrame):
        cols = df.columns
    else:
        cols = df
        is_list = True

    m_cols = {col: to_snake(col) for col in cols}

    if is_list:
        return list(m_cols.values())

    return df.pipe(lambda df: df.rename(columns=m_cols))


def parse_datecols(df, format=None):
    """Convert any columns with 'date' or 'time' in header name to datetime"""
    datecols = list(filter(lambda x: any(s in x.lower()
                    for s in ('date', 'time')), df.columns))
    df[datecols] = df[datecols].apply(
        pd.to_datetime, errors='coerce', format=format)
    return df


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


def df_proba(df, model, **kw):
    """Return df of predict_proba, with timestamp index"""
    arr = model.predict_proba(df)
    m = {-1: 'short', 0: 'neutral', 1: 'long'}
    cols = [f'proba_{m.get(c)}' for c in model.classes_]
    return pd.DataFrame(data=arr, columns=cols, index=df.index)


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
    # assert mase(np.array([1,2,3,4,5,6]),np.array([2,3,4,5,6,7])) == 1.0, "MASE bust"


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

    sm = sg.SignalManager(add_slope=5)

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


def proba_to_signal(s: pd.Series) -> pd.Series:
    """Convert probas btwn 0-1 to a signal of 0, 1 or -1 with threshold 0.5"""
    return np.sign(np.diff(np.sign(s - 0.5), prepend=np.array([0])))


def weighted_fit(name: str, n: int) -> dict:
    """Create dict of weighted samples for fit params"""
    return {f'{name}__sample_weight': np.linspace(0.5, 1, n)}
