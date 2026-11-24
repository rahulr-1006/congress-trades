from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_xml() -> bytes:
    return (FIXTURES / "sample_index.xml").read_bytes()
