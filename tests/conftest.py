# /home/sam069/projects/SportyDataFeeder/tests/conftest.py
#
# The test environment must be configured BEFORE any app module is imported:
# app.config.Settings is cached and app.database creates the engine at import
# time, so DATABASE_URL has to point at the throwaway SQLite file first.

import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(prefix="feeder_test_", suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
os.environ["SIMULATION_SPEED"] = "0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, Session, engine
from app.main import app


@pytest.fixture(autouse=True)
def fresh_db():
    from app.services.simulation import _simulations

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    _simulations.clear()
    yield
    Session.remove()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wait_for_simulation():
    """Poll the status endpoint until the simulation reaches a terminal state."""
    import time

    def _wait(client, match_id, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = client.get(f"/simulate/{match_id}/status")
            if response.status_code == 200:
                body = response.json()
                if body["status"] != "running":
                    return body
            time.sleep(0.02)
        raise TimeoutError(f"Simulation for match {match_id} did not finish within {timeout}s")

    return _wait
