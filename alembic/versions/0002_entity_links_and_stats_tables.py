"""entity_links, player_stats, match_predictions, player_match_ratings

Revision ID: 0002_entity_links_and_stats
Revises: 0001_initial_schema
Create Date: 2026-06-10 00:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_entity_links_and_stats"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entity_links",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("feeder_entity", sa.Text(), nullable=False),
        sa.Column("feeder_id", sa.Integer(), nullable=False),
        sa.Column("sporty_uuid", sa.Text(), nullable=False),
        sa.UniqueConstraint("feeder_entity", "feeder_id", name="uq_entity_links_feeder_entity_feeder_id"),
    )
    op.create_table(
        "player_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("gameweek", sa.Integer(), nullable=False),
        sa.Column("season", sa.String(), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("yellows", sa.Integer(), nullable=True),
        sa.Column("reds", sa.Integer(), nullable=True),
        sa.Column("points", sa.Float(), nullable=True),
        sa.Column("pts", sa.Integer(), nullable=True),
        sa.Column("ast", sa.Integer(), nullable=True),
        sa.Column("reb", sa.Integer(), nullable=True),
        sa.Column("stl", sa.Integer(), nullable=True),
        sa.Column("blk", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("player_id", "gameweek", "season", name="uq_player_stats_player_gameweek_season"),
    )
    op.create_table(
        "match_predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("home_win_prob", sa.Float(), nullable=False),
        sa.Column("draw_prob", sa.Float(), nullable=False),
        sa.Column("away_win_prob", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "player_match_ratings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("is_man_of_match", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("match_id", "player_id", name="uq_player_match_ratings_match_player"),
    )


def downgrade() -> None:
    op.drop_table("player_match_ratings")
    op.drop_table("match_predictions")
    op.drop_table("player_stats")
    op.drop_table("entity_links")
