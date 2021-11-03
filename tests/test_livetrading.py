from pytest import fixture

from jambot import getlog
from jambot import livetrading as live
from jambot.livetrading import ExchangeManager

log = getlog(__name__)


@fixture(scope='session')
def em() -> ExchangeManager:
    """Create ExchangeManager"""
    return ExchangeManager()


def test_iter_exchanges(em: ExchangeManager):
    """Test exchanges can be initialized"""
    exchs = list(em.iter_exchanges(refresh=True))
    log.info(f'Init {len(exchs)} exchanges.')


def test_run_strat_live(em: ExchangeManager):
    """Test strategy runs fully with all active exchanges
    - runs in test mode so only prints order output, doesn't send
    """
    live.run_strat_live(interval=15, em=em, test=True, use_test_models=True)
