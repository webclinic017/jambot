import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from seaborn import diverging_palette
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection._base import SelectorMixin
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             make_scorer, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from . import functions as f

# p_tmp = Path.home() / 'Desktop/sklearn_temp'
# print(p_tmp.exists())
_cmap = diverging_palette(240, 10, sep=10, n=21, as_cmap=True)

class ModelManager(object):
    """Manager class to perform cross val etc on multiple models with same underlying column transformer + data
    """    
    def __init__(self, ct=None, scoring=None, cv_args=None, **kw):
        random_state = 0
        cv_args = cv_args if not cv_args is None else {}
        df_results = pd.DataFrame()
        pipes = {}
        scores = {}
        models = {}
        grids = {}
        df_preds = {}
        set_self(vars())

        if any(item in kw for item in ('features', 'encoders')):
            self.ct = self.make_column_transformer(**kw)
    
    def make_column_transformer(self, features : dict, encoders : dict, **kw) -> ColumnTransformer:
        """Create ColumnTransformer from dicts of features and encoders

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
            transformers=[(name, model, features[name]) for name, model in encoders.items()])

        set_self(vars())
        return ct
    
    def show_ct(self, X_train : pd.DataFrame=None):
        X_train = X_train or getattr(self, 'X_train', None)
        
        if X_train is None:
            raise AttributeError('X_train not set!')
        
        data = self.ct.fit_transform(X_train)
        df_trans = df_transformed(data=data, ct=self.ct)
        print(df_trans.shape)
        display(df_trans.describe().T)

    def get_model(self, name : str, best_est=False):
        if best_est:
            return self.best_est(name=name)
        else:
            return self.models[name]

    def cross_val_feature_params(self, signal_manager, name, model, feature_params: dict):
        """Run full cross val pipe with single replacement of each feature in feature_params"""

        for df, param_name in signal_manager.replace_single_feature(df=self.df, feature_params=feature_params):
            
            # need to remake train/test splits every time
            X_train, y_train, X_test, y_test = self.make_train_test(
                df=df,
                target=['target'],
                train_size=0.9,
                shuffle=False)
            
            models = {f'{name}_{param_name}': model} # just remake models dict with modified param name as key
            self.cross_val(models=models, show=False)

        self.show()

    def cross_val(self, models : dict, show : bool=True, steps : list=None):
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

        for name, model in models.items():

            # allow passing model definition, or instantiated model
            if isinstance(model, type):
                model = model()

            model.random_state = self.random_state

            pipe = Pipeline(
                steps=[
                    ('ct', self.ct),
                    ('pca', PCA(n_components=10)),
                    (name, model)],
                # memory=str(p_tmp)
                )
            
            # insert extra steps in pipe, eg RFECV
            if not steps is None:
                if not isinstance(steps, list): steps = [steps]
                for step in steps:
                    pipe.steps.insert(step[0], step[1])
            
            # safe model/pipeline by name
            self.models[name] = model
            self.pipes[name] = pipe

            scores = cross_validate(pipe, self.X_train, self.y_train, **self.cv_args)
            self.scores[name] = scores
            self.df_results = self.df_results.pipe(append_mean_std_score, scores=scores, name=name)

        if show:
            self.show()
        
    def show(self):
        show_scores(self.df_results)
    
    def fit(self, name : str, best_est=False, model=None):
        """Fit model to training data"""
        if best_est:
            model = self.best_est(name)

        if model is None:
            model = self.pipes[name]

        model.fit(self.X_train, self.y_train.values.ravel())
        return model
    
    def y_pred(self, X, model=None, **kw):
        if model is None:
            model = self.fit(**kw)

        return model.predict(X)
    
    def class_rep(self, name : str = None, **kw):
        """Show classification report
        
        Parameters
        ----------
        name : str
            name of existing model
        """
        y_pred = self.y_pred(name=name, X=self.X_test, **kw)
        y_true = self.y_test.values.flatten()

        # classification report
        m = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(m).T
        display(df)
    
    def df_proba(self, df=None, model=None, **kw):
        """Return df of predict_proba, with timestamp index"""
        if df is None:
            df = self.X_test
        
        if model is None:
            model = self.fit(**kw)
            
        arr = model.predict_proba(df)
        m = {-1: 'short', 0: 'neutral', 1: 'long'}
        cols = [f'proba_{m.get(c)}' for c in model.classes_]
        return pd.DataFrame(data=arr, columns=cols, index=df.index)

    def add_proba(self, df, do=False, **kw):
        """Concat df of predict_proba"""
        return pd.concat([df, self.df_proba(**kw)], axis=1) if do else df

    def add_predict_iter(self, df, name, model, batch_size: int=96, min_size: int=180*24, max_train_size=None, regression=True):
        """Retrain model every x periods and add predictions for next batch_size"""
        df_train = df.copy()
        df = df.copy()

        df['y_pred'] = np.NaN
        if not regression:
            df['proba_long'] = np.NaN
            
        nrows = df.shape[0]
        num_batches = ((nrows - min_size) // batch_size) + 1

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
                y_train.values.ravel(),
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
            .assign(y_pred=self.y_pred(X=df.drop(columns=['target']), **kw)) \
            .pipe(self.add_proba, do=proba, **kw)
        
        # save predicted values for each model
        self.df_preds[kw['name']] = df

        return df
    
    def best_est(self, name : str):
        return self.grids[name].best_estimator_

    def search(self, name : str, params : dict, estimator=None, search_type : str='random', **kw):
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
        
        # rename params to include 'name__parameter'
        params = {f'{name}__{k}': v for k, v in params.items()}

        if estimator is None:
            estimator = self.pipes[name]
        
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
            .fit(self.X_train, self.y_train.values.ravel())
        
        self.grids[name] = grid

        results = {
            'Best params': grid.best_params_,
            'Best score': f'{grid.best_score_:.3f}'}

        pretty_dict(results)

        return grid

    def save_model(self, name : str, **kw):
        model = self.get_model(name=name, **kw)

        filename = f'{name}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    
    def load_model(self, name : str, **kw):
        filename = f'{name}.pkl'
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def make_train_test(self, df, target, train_size=0.8, **kw):
        """Make X_train, y_train etc from df
        
        Parameters
        ---------
        target : list   
            target column to remove for y_
        """
        df_train, df_test = train_test_split(df, train_size=train_size, random_state=self.random_state, **kw)

        def split(df):
            return df.drop(columns=target), df[target]
        
        X_train, y_train = split(df_train)
        X_test, y_test = split(df_test)

        set_self(vars())

        return X_train, y_train, X_test, y_test


def shap_explainer_values(X, y, ct, model, n_sample=2000):
    """Create shap values/explainer to be used with summary or force plot"""
    data = ct.fit_transform(X)
    X_enc = df_transformed(data=data, ct=ct)
    model.fit(X_enc, y.values.ravel())

    # use smaller sample to speed up plot
    X_sample = X_enc.sample(n_sample, random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample, X_enc

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

def format_cell(bg, t='black'):
    return f'background-color: {bg};color: {t};'

def bg(style, subset=None, rev=True):
    """Show style with highlights per column"""
    if subset is None:
        subset = style.data.columns

    cmap = _cmap.reversed() if rev else _cmap

    return style \
        .background_gradient(cmap=cmap, subset=subset, axis=0)

def highlight_val(df, m : dict):

    m_replace = {k: format_cell(bg=v[0], t=v[1]) for k, v in m.items()}

    return df.replace(m_replace)

def get_style(df):
    return df.style \
        .format('{:.3f}')

def show_scores(df):
    subset = [col for col in df.columns if any(item in col for item in ('test', 'train')) and not 'std' in col]

    style = get_style(df) \
        .pipe(bg, rev=False) \
        .pipe(bg, subset=subset)
    
    display(style)

def append_mean_std_score(df=None, scores=None, name=None, show=False):
    """Create df with mean and std of all scoring metrics"""
    if df is None:
        df = pd.DataFrame()

    if isinstance(name, Pipeline):
        # assume preprocessor then model name in pipeline.steps[1]
        name = name.steps[1][0]

    exclude = ['fit_time', 'score_time']
    name_cols = lambda cols, type_: {col: f'{type_}_{col}' for col in cols}
    score_cols = [col for col in scores.keys() if not col in exclude]
    mean_cols = name_cols(score_cols, 'mean')
    std_cols = name_cols(score_cols, 'std')

    df_scores = pd.DataFrame(scores).mean() \
        .rename(mean_cols) \
        .append(
            pd.DataFrame(scores) \
                .drop(columns=exclude) \
                .std() \
                .rename(std_cols))
    
    df.loc[name, df_scores.index] = df_scores
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
    cols = [col[1] for col in cols] # convert tuples back to list
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
        transformer_feature=get_ct_feature_names(ct), # this is a tuple of (transformer, feature_name)
        coef=coef)

    return pd.DataFrame(m) \
        .pipe(lambda df: pd.concat([
            df,
            pd.DataFrame(df['transformer_feature'].to_list())], axis=1)) \
        .rename(columns={0: 'transformer', 1: 'feature'}) \
        .drop(columns=['transformer_feature']) \
        .sort_values('coef', ascending=not best) \
        [:num_features] \
        .assign(coef=lambda x: x.coef.round(round_coef)) \
        [['transformer', 'feature', 'coef']]

def get_feature_out(estimator, feature_in):

    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return estimator.get_feature_names() # don't prepend 'vec'
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
        elif estimator=='passthrough':
            output_features.extend(make_tuple(name, ct._feature_names_in[features]))

    # print(output_features)      
    return output_features

def pretty_dict(m : dict, html=False, prnt=True) -> str:
    """Print pretty dict converted to newlines
    Paramaters
    ----
    m : dict\n
    html: bool
        Use <br> instead of html
    Returns
    -------
    str\n
        'Key 1: value 1\n
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
    
    if prnt:
        print(s)
    else:
        return s

def set_self(m, prnt=False, exclude=()):
    """Convenience func to assign an object's func's local vars to self"""
    if not isinstance(exclude, tuple): exclude = (exclude, )
    exclude += ('__class__', 'self') # always exclude class/self
    obj = m.get('self', None) # self must always be in vars dict

    if obj is None:
        return

    for k, v in m.items():
        if prnt:
            print(f'\n\t{k}: {v}')

        if not k in exclude:
            setattr(obj, k, v)
