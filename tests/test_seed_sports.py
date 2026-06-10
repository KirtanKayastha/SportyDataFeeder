# /home/sam069/projects/SportyDataFeeder/tests/test_seed_sports.py

from app.database import Session, Sport
from scripts.seed_sports import seed_sports


def test_seed_creates_both_sports(client):
    db = Session()
    try:
        created = seed_sports(db)
        assert sorted(created) == ["basketball", "football"]
        names = {sport.name for sport in db.query(Sport).all()}
        assert {"football", "basketball"} <= names
    finally:
        db.close()


def test_seed_is_idempotent(client):
    db = Session()
    try:
        seed_sports(db)
        assert seed_sports(db) == []
        assert db.query(Sport).count() == 2
    finally:
        db.close()
