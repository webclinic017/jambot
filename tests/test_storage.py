import logging
import warnings

from jambot.livetrading import ExchangeManager
from jambot.ml.storage import ModelStorageManager

# doesn't seem to work
warnings.filterwarnings('ignore', message='.*Failed to parse headers.*')
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented')

# azure storage logger logs everything to info by default
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').disabled = True


def test_fit_save(em: ExchangeManager):
    msm = ModelStorageManager(test=True)
    msm.fit_save_models(em=em)

    # check files uploaded to blob successfully
    saved_models = [p.name for p in msm.saved_models]
    assert sorted(saved_models) == sorted(msm.bs.list_files()), 'Saved models do not match files in azure blob'