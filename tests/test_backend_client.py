# /home/sam069/projects/SportyDataFeeder/tests/test_backend_client.py

import asyncio
import logging

import httpx
import pytest

from app.services.backend_client import BackendClient


def make_client(handler, backoff_base=0.0):
    return BackendClient(
        base_url="http://sporty.test",
        secret="test-secret",
        backoff_base=backoff_base,
        transport=httpx.MockTransport(handler),
    )


def run(coro):
    return asyncio.run(coro)


def test_successful_push_sends_secret_header():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["secret"] = request.headers.get("X-Feeder-Secret")
        return httpx.Response(200)

    client = make_client(handler)
    assert run(client.push_match_result({"sporty_match_id": "u1"})) is True
    assert seen["path"] == "/api/v1/feed/match-result"
    assert seen["secret"] == "test-secret"


def test_retries_then_succeeds(caplog):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    client = make_client(handler)
    with caplog.at_level(logging.WARNING):
        assert run(client.push_prediction({})) is True
    assert calls["n"] == 3
    assert "retrying" in caplog.text


def test_exhausted_retries_log_error_and_return_false(caplog):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("backend down")

    client = make_client(handler)
    with caplog.at_level(logging.ERROR):
        assert run(client.push_player_ratings({})) is False
    assert calls["n"] == 3
    assert "failed after 3 attempts" in caplog.text


def test_http_4xx_is_not_success():
    def handler(request):
        return httpx.Response(401)

    client = make_client(handler)
    assert run(client.push_match_result({})) is False


def test_backoff_grows_exponentially(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(request):
        return httpx.Response(500)

    client = make_client(handler, backoff_base=1.5)
    assert run(client.push_match_result({})) is False
    assert sleeps == [1.5, 1.5**2]
