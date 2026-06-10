# /home/sam069/projects/SportyDataFeeder/tests/test_links.py

from app.database import Session
from app.services.links import get_sporty_uuid

UUID_A = "11111111-1111-1111-1111-111111111111"
UUID_B = "22222222-2222-2222-2222-222222222222"


def test_post_links_creates_mapping(client):
    response = client.post(
        "/links",
        json={"feeder_entity": "match", "feeder_id": 7, "sporty_uuid": UUID_A},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["feeder_entity"] == "match"
    assert body["feeder_id"] == 7
    assert body["sporty_uuid"] == UUID_A


def test_post_links_upserts_on_conflict(client):
    first = client.post(
        "/links",
        json={"feeder_entity": "player", "feeder_id": 3, "sporty_uuid": UUID_A},
    )
    second = client.post(
        "/links",
        json={"feeder_entity": "player", "feeder_id": 3, "sporty_uuid": UUID_B},
    )
    assert second.status_code == 200
    assert second.json()["sporty_uuid"] == UUID_B
    assert second.json()["id"] == first.json()["id"]

    links = client.get("/links", params={"feeder_entity": "player"}).json()
    assert len(links) == 1


def test_post_links_rejects_unknown_entity(client):
    response = client.post(
        "/links",
        json={"feeder_entity": "stadium", "feeder_id": 1, "sporty_uuid": UUID_A},
    )
    assert response.status_code == 422


def test_get_sporty_uuid_helper(client):
    client.post(
        "/links",
        json={"feeder_entity": "team", "feeder_id": 42, "sporty_uuid": UUID_A},
    )
    db = Session()
    try:
        assert get_sporty_uuid(db, "team", 42) == UUID_A
        assert get_sporty_uuid(db, "team", 99) is None
    finally:
        db.close()
