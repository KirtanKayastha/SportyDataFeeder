# /home/sam069/projects/SportyDataFeeder/tests/test_smoke.py


def test_app_boots_and_health_is_open(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"


def test_settings_come_from_environment():
    from app.config import get_settings

    settings = get_settings()
    assert settings.DATABASE_URL.startswith("sqlite:///")
    assert settings.SIMULATION_SPEED == 0
