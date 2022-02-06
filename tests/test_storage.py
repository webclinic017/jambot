import logging

from pytest import fixture, mark

from jambot.livetrading import ExchangeManager
from jambot.ml.storage import ModelStorageManager
from jambot.tradesys.symbols import Symbol

# azure storage logger logs everything to info by default
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').disabled = True


@fixture(scope='session')
def test_fit_save(em: ExchangeManager, symbol: Symbol):
    msm = ModelStorageManager(test=True)
    msm.fit_save_models(em=em, overwrite_all=True, symbol=symbol)

    # check files uploaded to blob successfully
    assert msm.local_model_names(symbol=symbol) == msm.bs.list_files(
        match=symbol), 'Saved models do not match files in azure blob'


@mark.usefixtures('test_fit_save')
def _test_fit_save():
    """Just call fixture"""
    return
