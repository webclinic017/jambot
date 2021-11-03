"""Define pytest fixtures and configuration
"""

import os
from pathlib import Path

import pytest
from discord import Webhook
from pytest import fixture, mark, raises  # noqa


def set_fixture_modules():
    """Make all fixtures available from any files"""
    p_test = Path(__file__).parent

    def as_module(p: Path) -> str:
        """Convert path obj to module-like str"""
        return p.relative_to(p_test.parent).as_posix().replace('/', '.').replace('.py', '')

    exclude = ('__', 'conftest')
    return [as_module(p) for p in p_test.rglob('*.py') if not any(item in str(p) for item in exclude)]


@fixture(autouse=True)
def tests_setup_and_teardown(monkeypatch):
    """Set env var so everything can check for test environment"""
    monkeypatch.setenv('pytest', '1')
    monkeypatch.setattr(Webhook, 'send', lambda *args: print(*args))


pytest_plugins = set_fixture_modules()

skip_github = mark.skipif(
    'GITHUB_ACTIONS' in os.environ,
    reason='Can\'t run this test on github.')


def pytest_addoption(parser):
    """Run slow tests if --runslow command used"""
    parser.addoption(
        '--runslow',
        action='store_true',
        default=False,
        help='run slow tests')

    parser.addoption(
        '--exch_name',
        action='store',
        default='bitmex',
        choices=('bybit', 'bitmex'))


@fixture(scope='session')
def exch_name(request):
    return request.config.getoption('--exch_name')


def pytest_collection_modifyitems(config, items):
    """Check if --runslow command used"""

    # --runslow used, don't skip slow tests
    if config.getoption('--runslow'):
        return

    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
