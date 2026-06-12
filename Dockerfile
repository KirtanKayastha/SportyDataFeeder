FROM python:3.12-slim

WORKDIR /srv/feeder

# psycopg2-binary ships its own libpq; no system build deps needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY alembic/ alembic/
COPY app/ app/
COPY scripts/ scripts/

# models_pkl/ is intentionally not copied: pkl files are environment-specific
# (event_rates is keyed by this DB's player ids). Mount it as a volume or
# train inside the container with `python -m scripts.train_models`.

EXPOSE 8000

# Migrations run at container start so a fresh database is usable immediately;
# alembic upgrade head is a no-op when the schema is current.
CMD ["sh", "-c", "alembic upgrade head && python -m scripts.seed_sports && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
