# /home/sam069/projects/SportyDataFeeder/tests/test_migrations.py

import os
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.database import Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_alembic_upgrade_head_builds_full_schema_on_empty_db():
    db_fd, db_path = tempfile.mkstemp(prefix="feeder_migration_test_", suffix=".db")
    os.close(db_fd)
    url = f"sqlite:///{db_path}"
    try:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        config.set_main_option("sqlalchemy.url", url)
        command.upgrade(config, "head")

        engine = create_engine(url)
        try:
            migrated_tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
        finally:
            engine.dispose()

        model_tables = set(Base.metadata.tables)
        assert migrated_tables == model_tables
    finally:
        os.unlink(db_path)
