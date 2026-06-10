# /home/sam069/projects/SportyDataFeeder/app/services/links.py

from app.database import EntityLink


def upsert_link(db, feeder_entity: str, feeder_id: int, sporty_uuid: str) -> EntityLink:
    link = (
        db.query(EntityLink)
        .filter_by(feeder_entity=feeder_entity, feeder_id=feeder_id)
        .first()
    )
    if link:
        link.sporty_uuid = sporty_uuid
    else:
        link = EntityLink(feeder_entity=feeder_entity, feeder_id=feeder_id, sporty_uuid=sporty_uuid)
        db.add(link)
    db.commit()
    db.refresh(link)
    return link


def get_sporty_uuid(db, entity: str, feeder_id: int) -> str | None:
    link = (
        db.query(EntityLink)
        .filter_by(feeder_entity=entity, feeder_id=feeder_id)
        .first()
    )
    return link.sporty_uuid if link else None
