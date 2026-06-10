"""add event_id idempotency column to events

Revision ID: 0003_add_event_id
Revises: 0002_entity_links_and_stats
Create Date: 2026-06-10 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_add_event_id"
down_revision = "0002_entity_links_and_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(sa.Column("event_id", sa.String(), nullable=True))
        batch_op.create_unique_constraint("uq_events_event_id", ["event_id"])


def downgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_constraint("uq_events_event_id", type_="unique")
        batch_op.drop_column("event_id")
