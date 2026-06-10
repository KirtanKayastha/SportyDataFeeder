# /home/sam069/projects/SportyDataFeeder/scripts/seed_sports.py
#
# Idempotently seeds the sports table with the rows the importers and the
# simulator expect ("football", "basketball"). Run after migrations:
#   alembic upgrade head && python -m scripts.seed_sports

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import Session, Sport

logger = logging.getLogger(__name__)

SPORT_NAMES = ["football", "basketball"]


def seed_sports(db) -> list[str]:
    """Insert any missing sport rows. Returns the names that were created."""
    created = []
    for name in SPORT_NAMES:
        existing = db.query(Sport).filter(Sport.name.ilike(name)).first()
        if existing:
            logger.info("Sport '%s' already present (id=%s)", existing.name, existing.id)
            continue
        db.add(Sport(name=name))
        created.append(name)
    if created:
        db.commit()
        logger.info("Seeded sports: %s", ", ".join(created))
    else:
        logger.info("Sports table already seeded; nothing to do")
    return created


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = Session()
    try:
        seed_sports(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
