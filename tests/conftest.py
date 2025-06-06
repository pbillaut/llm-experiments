from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource() -> Path:
    return Path(__file__).parent / "resources"
