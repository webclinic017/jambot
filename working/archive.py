
def fit_save_models(
        self,
        em: 'ExchangeManager',
        # df: pd.DataFrame,
        symbol: Union[Symbol, str],
        name: str = NAME,
        interval: int = 15,
        overwrite_all: bool = False) -> None:
    """Retrain single new, or overwrite all models
    - run live every n hours (24)
    - models trained AT 18:00, but only UP TO eg 10:00
    - all models start fit from same d_lower

    Parameters
    ----------
    em : ExchangeManager
    df : pd.DataFrame, optional
        df with OHLC from db, default None
    symbol : Union[Symbol, str]
        eg Symbol('XBTUSD', 'bitmex') or 'MULTI_ALTS'
    name : str, optional
        model name, default 'lgbm'
    interval : int, optional
        default 15
    overwrite_all : bool, optional
        If True delete all existing models and retrain,
        else retrain newest only, default False
    interval : int, optional
        default 15
    """

    dask.config.set(scheduler='synchronous')

    # TODO - keep 3 trained models saved
    # - use most recent to train new model, then drop oldest
    # if no models exists, train from beginning, else train on data newer than model date
    # need to figure out wm filter_fit... download ALL OHLC data...? only need close col at least

    self.bs.download_dir(p=self.p_model, mirror=True, match=symbol)

    # check to make sure all models downloaded
    if self.n_models_local(symbol) < self.n_models or overwrite_all:
        overwrite_all = True
        startdate = self.d_lower
    else:
        offset = {1: 16, 15: 6}[interval]  # get days offset
        startdate = f.inter_now(interval) + delta(days=-offset)

    # azure limited to 1.5gb memory
    if cf.AZURE_WEB and overwrite_all:
        raise RuntimeError('Cannot overwrite_all on azure.')

    log.info(f'fit_save_models: symbol={symbol}, overwrite={overwrite_all}, startdate={startdate}')

    n_periods = cf.dynamic_cfg(symbol=symbol, keys='target_n_periods')

    # FIXME temp solution
    exch_name = symbol.exch_name if not symbol == cf.MULTI_ALTS else 'bybit'

    # FIXME funding_exch
    # won't need to download all data necessarily
    df = self.dm.get_df(
        symbols={exch_name: symbol},
        startdate=startdate,
        interval=interval,
        # funding_exch=em.default('bitmex'),
        db_only=True,
        # local_only=not cf.AZURE_WEB
    )

    # laod df_close full history (~12s, 16mb, >1.25M rows)
    # TODO figure out how to cache the quantile, or compute from db?
    df_close = self.dm.get_df(
        symbols={exch_name: symbol},
        startdate=self.d_lower,
        interval=interval,
        db_only=True,
        close_only=True)

    wm = WeightsManager.from_config(df=df_close, symbol=symbol)
    # df['weights'] = wm.weights

    # df_ohlc = df
    # ddf = None
    # set back @cut_hrs due to losing @n_periods for training preds
    cut_hrs = math.ceil(n_periods / {1: 1, 15: 4}[self.interval])
    reset_hour_offset = self.reset_hour - cut_hrs  # 18
    # print('cut_hrs:', cut_hrs)  # 8
    # print('reset_hour_offset:', reset_hour_offset)  # 10

    # max date where hour is greater or equal to 18:00
    d_upper = f.date_to_dt(
        df.query('timestamp.dt.hour >= @reset_hour_offset')
        .index.get_level_values('timestamp').max().date()) \
        + delta(hours=reset_hour_offset)  # get date only eg '2022-01-01' then add delta hoursg
    # print('d_upper:', d_upper)  # 2022-01-17 10:00:00

    filter_quantile = cf.dynamic_cfg(symbol=symbol, keys='filter_fit_quantile')

    # add signals per symbol group, drop last targets
    # TODO add funding_exch: funding_exch=em.default('bitmex')
    df = df \
        .pipe(
            md.add_signals,
            name=name,
            symbol=symbol,
            use_important_dynamic=True,
            drop_ohlc=True,
            drop_target_periods=n_periods) \
        .pipe(
            wm.filter_quantile,
            quantile=filter_quantile,
            _log=False)

    # set up cluster and workers
    # client = Client(n_workers=1, threads_per_worker=1, processes=False)  # memory_limit='2GB'
    # batch_size = 50000

    # add signals to grouped individual symbols
    # for _symbol, _df in df.groupby('symbol'):
    #     print(_symbol, _df.shape)

    #     # keep weights with df for sample_weights
    #     # AND filter by highest quantile
    #     _df = _df \
    #         .pipe(
    #             md.add_signals,
    #             name=name,
    #             drop_ohlc=False,
    #             use_important_dynamic=True,
    #             symbol=symbol,
    #             drop_target_periods=n_periods) \
    #         .pipe(
    #             wm.filter_quantile,
    #             _log=False,
    #             quantile=filter_quantile) \
    #         .pipe(pu.safe_drop, cols=cf.DROP_COLS)

    # droplevel copies df
    # _df.set_index(_df.index.get_level_values('timestamp'), inplace=True)

    # if ddf is None:
    #     ddf = da.dataframe.from_pandas(_df, chunksize=batch_size)
    # else:
    #     ddf = ddf.append(_df)

    # remove last _df from memory before train
    # del wm, _df, df
    # gc.collect()

    # ddf = ddf \
    #     .reset_index(drop=False) \
    #     .set_index('timestamp')

    # return

    # if self.d_latest_model(symbol=symbol) < d_upper or overwrite_all:
    #     self.clean(symbol=symbol, last_only=not overwrite_all)
    if overwrite_all:
        self.clean(symbol=symbol, last_only=not overwrite_all)

    cut_mins = {1: 60, 15: 15}[self.interval]  # to offset .loc[:d] for each prev model/day
    model = md.make_model(name, symbol=symbol)
    # n_models = 1 if not overwrite_all else self.n_models
    n_models = self.n_models

    # treat d_upper (max in df) as current date
    # if no models exist, or overwrite_all, train first model up to earliest date
    # - then use init_model
    # else, use latest model
    keep_models = []  # type: List[Path]

    # loop models from oldest > newest (2, 1, 0)
    for i in range(n_models - 1, -1, -1):
        log.warning(i)
        delta_mins = -i * cut_mins * self.batch_size_cdls
        d = d_upper + delta(minutes=delta_mins)  # cut 0, 24, 48 hrs
        d_cur_model = d + delta(hours=cut_hrs)
        d_prev = d + delta(hours=-self.batch_size)  # prev model date
        d_prev_model = d_prev + delta(hours=cut_hrs)
        print('d:', d, 'd_prev:', d_prev, 'd_prev_model', d_prev_model)

        # do we have a previous model
        p_prev = self.get_model_path(symbol, d_prev_model)
        p_cur = self.get_model_path(symbol, d_cur_model)
        keep_models.append(p_cur)

        print('p_prev:', p_prev.name)
        print('p_cur:', p_cur.name)

        is_first = i == n_models - 1
        if not p_cur.exists() or (is_first and overwrite_all):

            # use df filtered to dates greater than last trained model
            if is_first:
                init_model = None
                d_lower = df.index.get_level_values('timestamp').min()
            else:
                init_model = self.load_model(p=p_prev)
                d_lower = d_prev + f.inter_offset(interval)

            print(f'rng: {d_lower} - {d}')

            idx_slice = pd.IndexSlice[:, d_lower: d]
            x_train, y_train = sk.split(df.loc[idx_slice, :])

            log.info(
                f'fitting model: x_train.shape={x_train.shape}, is_first={is_first}, init_model={init_model}')

            init_model = model.fit(
                X=x_train,
                y=y_train,
                sample_weight=wm.weights.loc[x_train.index],
                init_model=init_model,
            )

            # save - add back cut hrs so always consistent
            self.save_model(model=model, symbol=symbol, d=d_cur_model)
        else:
            log.info('model exists, not overwriting.')

        # trim df to older dates by cutting off progressively larger slices
        # for i in range(n_models):
        #     log.info(f'model num: {i}')

        #     delta_mins = -i * cut_mins * self.batch_size_cdls
        #     d = d_upper + delta(minutes=delta_mins)
        #     # print('delta_mins:', delta_mins)
        #     # print('d:', d)  # 2022-01-17 10:00
        #     init_model = None

        #     for j, _df in enumerate(ddf.loc[:d].partitions):
        #         log.info(f'fitting model: {j}')
        #         # train UP TO eg 2022-01-17 10:00 UTC
        #         x_train, y_train = sk.split(
        #             # df=df.loc[pd.IndexSlice[:, :d], :]
        #             # df=df[:d]
        #             # df=ddf.loc[:d]
        #             df=_df.compute()
        #         )
        #         del _df

        #         # fit - using weighted currently
        #         # model.fit(x_train, y_train, sample_weight=wm.weights.loc[x_train.index.compute(), :])
        #         # model.fit(x_train.drop(columns=['weights']), y_train, sample_weight=x_train.weights)

        #         init_model = model.fit(
        #             X=x_train.drop(columns=['weights']),
        #             y=y_train,
        #             sample_weight=x_train.weights,
        #             init_model=init_model,
        #             # batch_size=batch_size
        #         )

        # client.close()

    # delete oldest model
    for p in self.local_models(symbol=symbol):
        if not p in keep_models:
            p.unlink()

    # mirror saved models to azure blob
    self.bs.upload_dir(p=self.p_model, mirror=True, match=symbol)


def _fit_save_models(
    self,
    # em: 'ExchangeManager',
    # df: pd.DataFrame,
    symbol: Union[Symbol, str],
    name: str = NAME,
    interval: int = 15,
        overwrite_all: bool = False) -> None:
    """Retrain single new, or overwrite all models
    - run live every n hours (24)
    - models trained AT 18:00, but only UP TO eg 10:00

    Parameters
    ----------
    em : ExchangeManager
    df : pd.DataFrame, optional
        df with OHLC from db, default None
    symbol : Union[Symbol, str]
        eg Symbol('XBTUSD', 'bitmex') or 'MULTI_ALTS'
    name : str, optional
        model name, default 'lgbm'
    interval : int, optional
        default 15
    overwrite_all : bool, optional
        If True delete all existing models and retrain,
        else retrain newest only, default False
    interval : int, optional
        default 15
    """
    self.bs.download_dir(p=self.p_model, mirror=True, match=symbol)

    # check to make sure all models downloaded
    if self.n_models_local(symbol) < self.n_models or overwrite_all:
        overwrite_all = True
        startdate = self.d_lower  # download full OHLC history
    else:
        offset = {1: 16, 15: 6}[interval]  # get days offset
        startdate = f.inter_now(interval) + delta(days=-offset)  # only download last 6 days

    # azure limited to 1.5gb memory
    if cf.AZURE_WEB and overwrite_all:
        raise RuntimeError('Cannot overwrite_all on azure.')

    log.info(f'fit_save_models: symbol={symbol}, overwrite={overwrite_all}, startdate={startdate}')

    n_periods = cf.dynamic_cfg(symbol=symbol, keys='target_n_periods')

    # FIXME temp solution
    exch_name = symbol.exch_name if not symbol == cf.MULTI_ALTS else 'bybit'

    # FIXME funding_exch
    # won't need to download all data necessarily
    df = self.dm.get_df(
        symbols={exch_name: symbol},
        startdate=startdate,
        interval=interval,
        # funding_exch=em.default('bitmex'),
        db_only=True,
        # local_only=not cf.AZURE_WEB
    )

    # laod df_close full history (~12s, 16mb, >1.25M rows)
    # TODO figure out how to cache the quantile, or compute from db?
    df_close = self.dm.get_df(
        symbols={exch_name: symbol},
        startdate=self.d_lower,
        interval=interval,
        db_only=True,
        close_only=True)

    wm = WeightsManager.from_config(df=df_close, symbol=symbol)

    # set back @cut_hrs due to losing @n_periods for training preds
    cut_hrs = math.ceil(n_periods / {1: 1, 15: 4}[self.interval])
    reset_hour_offset = self.reset_hour - cut_hrs  # 18
    # print('cut_hrs:', cut_hrs)  # 8
    # print('reset_hour_offset:', reset_hour_offset)  # 10

    # max date where hour is greater or equal to 18:00
    d_upper = f.date_to_dt(
        df.query('timestamp.dt.hour >= @reset_hour_offset')
        .index.get_level_values('timestamp').max().date()) \
        + delta(hours=reset_hour_offset)  # get date only eg '2022-01-01' then add delta hoursg
    # print('d_upper:', d_upper)  # 2022-01-17 10:00:00

    filter_quantile = cf.dynamic_cfg(symbol=symbol, keys='filter_fit_quantile')

    # add signals per symbol group, drop last targets
    # TODO add funding_exch: funding_exch=em.default('bitmex')
    df = df \
        .pipe(
            md.add_signals,
            name=name,
            symbol=symbol,
            use_important_dynamic=True,
            drop_ohlc=True,
            drop_target_periods=n_periods) \
        .pipe(
            wm.filter_quantile,
            quantile=filter_quantile,
            _log=False)

    # delete oldest model or all models
    if self.d_latest_model(symbol=symbol) < d_upper or overwrite_all:
        self.clean(symbol=symbol, last_only=not overwrite_all)

    cut_mins = {1: 60, 15: 15}[self.interval]  # to offset .loc[:d] for each prev model/day
    model = md.make_model(name, symbol=symbol)
    n_models = self.n_models

    # loop models from oldest > newest
    for i in range(n_models - 1, -1, -1):
        # log.warning(i)
        delta_mins = -i * cut_mins * self.batch_size_cdls
        d = d_upper + delta(minutes=delta_mins)  # cut 0, 24, 48 hrs
        d_cur_model = d + delta(hours=cut_hrs)
        d_prev = d + delta(hours=-self.batch_size)  # prev model date
        d_prev_model = d_prev + delta(hours=cut_hrs)
        # print('d:', d, 'd_prev:', d_prev, 'd_prev_model', d_prev_model)

        # do we have a previous model
        p_prev = self.get_model_path(symbol, d_prev_model)
        p_cur = self.get_model_path(symbol, d_cur_model)
        print('p_prev:', p_prev.name)
        print('p_cur:', p_cur.name)

        is_first = i == n_models - 1
        if not p_cur.exists() or (is_first and overwrite_all):

            # use df filtered to dates greater than last trained model
            if is_first:
                init_model = None
                d_lower = df.index.get_level_values('timestamp').min()
            else:
                init_model = self.load_model(p=p_prev)
                d_lower = d_prev + f.inter_offset(interval)
            print(f'rng: {d_lower} - {d}')

            idx_slice = pd.IndexSlice[:, d_lower: d]
            x_train, y_train = pu.split(df.loc[idx_slice, :])
            log.info(
                f'fitting model: x_train.shape={x_train.shape}, is_first={is_first}, init_model={init_model}')

            init_model = model.fit(
                X=x_train,
                y=y_train,
                sample_weight=wm.weights.loc[x_train.index],
                init_model=init_model)

            # save - add back cut hrs so always consistent
            self.save_model(model=model, symbol=symbol, d=d_cur_model)
        else:
            log.info('model exists, not overwriting.')

    # mirror saved models to azure blob
    self.bs.upload_dir(p=self.p_model, mirror=True, match=symbol)


class DaskSequence(Sequence):
    def __init__(self, ddf: 'DDF', batch_size: int = 10000):
        """
        Construct a sequence object from HDF5 with required interface.
        Parameters
        ----------
        ddf : dask.dataframe
        batch_size : int
            Size of a batch. When reading data to construct lightgbm Dataset, each read reads batch_size rows.
        """
        # We can also open HDF5 file once and get access to

        # reset index of dask df will be 0...n PER PARTITION
        self.ddf = ddf.reset_index(drop=True) \
            .pipe(monotonic_index) \
            .pipe(repartition, batch_size=batch_size)

        self._len = len(self.ddf)

        self.batch_size = batch_size
        self.active_block = pd.DataFrame()
        self.active_block_idx = -1

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """Get values from ddf as computed np array vals
        - NOTE cannot call range index which overlaps block size eg 99_999:100_001

        Parameters
        ----------
        idx : Union[int, slice]
            eg 3 or 10004:10032

        Returns
        -------
        np.ndarray
            1d or 2d np array
        """

        # find block number that index intersects (int or slice)
        if not isinstance(idx, slice):
            i_block = idx // self.batch_size
            # offset = self.batch_size * i_block
            # idx = idx - offset
        else:
            i_block = idx.start // self.batch_size
            # offset = self.batch_size * i_block
            # idx = slice(idx.start - offset, idx.stop - offset)

        # load new block
        if not i_block == self.active_block_idx:
            self.active_block = self.ddf.partitions[i_block].compute()  # type: pd.DataFrame
            self.active_block_idx = i_block
            print(f'active_block: {i_block}')

        # lgbm dataset from samples needs values to be np.float64
        return self.active_block.loc[idx].values.astype(np.float64)

    def __len__(self):
        return self._len


def bad_add_predict_iter(
        df: pd.DataFrame,
        wm: WeightsManager,
        model: 'LGBMClsLog',
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
    model : LGBMClsLog
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

    # convert months to periods  NOTE not implemented for date indexing yet
    # if not max_train_size is None:
    #     max_train_size = max_train_size * 30 * 24 * {1: 1, 15: 4}[interval]

    df = df.pipe(pu.safe_drop, cf.DROP_COLS)

    init_model = None
    d_train_lower = None
    d_train_upper = None
    dfs = []

    # for i in tqdm(range(num_batches), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    for i in range(num_batches):
        # fit up to current test batch, predict on test batch
        # d_train_lower : d_train_upper : d_test_upper

        d_train_lower = d_start if d_train_upper is None else d_train_upper

        d_train_upper = split_date + delta(minutes=i * batch_size * delta_mins)
        # d_lower = split_date + delta(minutes=i * batch_size * delta_mins)
        d_test_upper = d_train_upper + delta(minutes=(batch_size - 1) * delta_mins)

        idx_train_slice = pd.IndexSlice[:, d_train_lower: d_train_upper + delta(minutes=-delta_mins)]
        df_train = df.loc[idx_train_slice, :]

        idx_test_slice = pd.IndexSlice[:, d_train_upper: d_test_upper]

        if filter_fit_quantile:
            # TODO possibly filter highest PER SYMBOL, not all combined
            df_train = wm.filter_quantile(df_train, quantile=filter_fit_quantile, _log=False)

        # log.info(f'{i}: shape={df_train.shape}, {d_train_lower} > {d_train_upper} > {d_test_upper}')

        x_train, y_train = pu.split(df_train)
        x_test, _ = pu.split(df.loc[idx_test_slice, :])

        init_model = model.fit(
            x_train,
            y_train,
            init_model=init_model,
            **wm.fit_params(x_train))

        with open(f'new_model_{i}.txt', 'w') as file:
            file.write(init_model.booster_.model_to_string())

        # if len(idx_test_slice) == 0:
        #     return None
        # else:
        proba_long = md.df_proba(x=x_test, model=init_model)['proba_long']

        pd.DataFrame(
            data=dict(
                y_pred=init_model.predict(x_test),
                proba_long=proba_long)) \
            .pipe(pu.append_list, dfs)

    # par = ProgressParallel(batch_size=2, n_jobs=n_jobs, total=num_batches)
    # result = par(delayed(_fit)(i=i) for i in range(num_batches))
    # result = [_fit(i) for i in range(num_batches)]

    return df_orig.pipe(pu.left_merge, pd.concat(dfs))
