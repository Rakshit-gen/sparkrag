import pytest

from sparkrag.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark("sparkrag-tests")
    yield session
    session.stop()
