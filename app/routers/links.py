# /home/sam069/projects/SportyDataFeeder/app/routers/links.py

from fastapi import APIRouter, Depends, Query

from app.database import EntityLink, get_db
from app.schemas import EntityLinkCreate, EntityLinkRead
from app.services.links import upsert_link

router = APIRouter()


@router.get("/links", response_model=list[EntityLinkRead])
def list_links(feeder_entity: str | None = Query(default=None), db=Depends(get_db)):
    query = db.query(EntityLink)
    if feeder_entity is not None:
        query = query.filter(EntityLink.feeder_entity == feeder_entity)
    return query.order_by(EntityLink.feeder_entity.asc(), EntityLink.feeder_id.asc()).all()


@router.post("/links", response_model=EntityLinkRead)
def create_or_update_link(payload: EntityLinkCreate, db=Depends(get_db)):
    return upsert_link(db, payload.feeder_entity, payload.feeder_id, payload.sporty_uuid)
