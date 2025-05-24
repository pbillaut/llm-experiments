from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def resources_root() -> Path:
    return Path(__file__).parent / 'resources'
