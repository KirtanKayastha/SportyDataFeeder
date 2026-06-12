# R-2.10: every route requires X-Feeder-Secret except /health, /docs, /openapi.json.

from fastapi.testclient import TestClient

from app.main import app


def _anonymous_client():
    return TestClient(app)


def test_health_is_open_without_secret(client):
    with _anonymous_client() as anon:
        response = anon.get("/health")
    assert response.status_code == 200


def test_docs_and_openapi_are_open_without_secret():
    with _anonymous_client() as anon:
        assert anon.get("/docs").status_code == 200
        assert anon.get("/openapi.json").status_code == 200


def test_protected_route_without_secret_is_401(client):
    with _anonymous_client() as anon:
        response = anon.get("/sports/")
    assert response.status_code == 401
    assert "X-Feeder-Secret" in response.json()["detail"]


def test_protected_route_with_wrong_secret_is_401(client):
    with _anonymous_client() as anon:
        response = anon.get("/sports/", headers={"X-Feeder-Secret": "wrong-secret"})
    assert response.status_code == 401


def test_protected_route_with_correct_secret_succeeds(client):
    response = client.get("/sports/")
    assert response.status_code == 200


def test_unknown_path_still_requires_secret():
    # 401 must win over 404 so the middleware never leaks route existence.
    with _anonymous_client() as anon:
        response = anon.get("/definitely-not-a-route")
    assert response.status_code == 401
