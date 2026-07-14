# /home/sam069/projects/SportyDataFeeder/app/services/backend_client.py
#
# Outbound push to the Sporty backend (PRD R-4.2). httpx async over a
# persistent, connection-pooled client (see get_backend_client), X-Feeder-Secret
# header, 3 attempts with exponential backoff (1.5^n seconds). A failed push is
# never fatal: it logs ERROR and returns False — events persist locally and can
# be re-sent via POST /matches/{id}/replay-push.

import asyncio
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE = 1.5
REQUEST_TIMEOUT = 5.0
# Orchestration calls (schedule/register/demo-setup) do several DB writes and
# can hit a cold remote database, so they get a generous timeout.
SETUP_TIMEOUT = 30.0

MATCH_RESULT_PATH = "/api/v1/feed/match-result"
PREDICTION_PATH = "/api/v1/feed/prediction"
MODEL_METRICS_PATH = "/api/v1/feed/model-metrics"
PLAYER_RATINGS_PATH = "/api/v1/feed/player-ratings"
MATCH_LINEUPS_PATH = "/api/v1/feed/match-lineups"
SCHEDULE_MATCH_PATH = "/api/v1/feed/schedule-match"
REGISTER_PLAYERS_PATH = "/api/v1/feed/register-players"
RESOLVE_PLAYERS_PATH = "/api/v1/feed/resolve-players"
DEMO_SETUP_PATH = "/api/v1/feed/demo-setup"


def feeder_match_external_ref(match_id: int) -> str:
    """The external_ref every schedule_match caller must pass. Without one,
    the backend derives an identity hash from home_team|away_team|match_date
    (feed.py's ScheduleMatchPayload default) — fine for a single push, but it
    means two DIFFERENT feeder matches sharing a team pairing and date (e.g. a
    replayed demo fixture) alias onto the SAME Sporty match/Redis channel, so
    a second simulation run overwrites/cross-talks with the first instead of
    getting its own match. Keying on the feeder's own match id guarantees
    every feeder match gets a distinct Sporty match, while still being
    idempotent for repeat calls about the SAME feeder match id."""
    return f"feeder:match:{match_id}"


class BackendClient:
    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
        backoff_base: float = BACKOFF_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.SPORTY_BACKEND_URL).rstrip("/")
        self.secret = secret if secret is not None else settings.FEEDER_SECRET
        self.backoff_base = backoff_base
        # One client (and its connection pool) for the instance's whole life —
        # a match simulation pushes dozens of events through the same
        # BackendClient, and each call used to open+close a fresh connection.
        self._client = httpx.AsyncClient(transport=transport, timeout=REQUEST_TIMEOUT)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict) -> bool:
        url = f"{self.base_url}{path}"
        headers = {"X-Feeder-Secret": self.secret}
        last_error: str = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                if response.status_code < 400:
                    logger.info("Pushed %s (attempt %s)", path, attempt)
                    return True
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)

            if attempt < MAX_ATTEMPTS:
                delay = self.backoff_base**attempt
                logger.warning(
                    "Push to %s failed (attempt %s/%s: %s); retrying in %.2fs",
                    url, attempt, MAX_ATTEMPTS, last_error, delay,
                )
                await asyncio.sleep(delay)

        logger.error(
            "Push to %s failed after %s attempts (%s); continuing — events persist locally for replay",
            url, MAX_ATTEMPTS, last_error,
        )
        return False

    async def _post_json(self, path: str, payload: dict) -> dict:
        """POST and return the parsed JSON response (single attempt; used by the
        demo orchestration where the caller needs the response body, not just a
        delivered/failed flag). Raises on HTTP/transport error."""
        url = f"{self.base_url}{path}"
        headers = {"X-Feeder-Secret": self.secret}
        response = await self._client.post(url, json=payload, headers=headers, timeout=SETUP_TIMEOUT)
        response.raise_for_status()
        return response.json()

    async def _delete_json(self, path: str) -> dict:
        """DELETE and return the parsed JSON response. Single attempt; raises on
        HTTP/transport error so the caller sees 404 (unknown) / 409 (live)."""
        url = f"{self.base_url}{path}"
        headers = {"X-Feeder-Secret": self.secret}
        response = await self._client.delete(url, headers=headers, timeout=SETUP_TIMEOUT)
        response.raise_for_status()
        return response.json()

    async def schedule_match(self, payload: dict) -> dict:
        """Register a simulated fixture on the backend → returns sporty_match_id."""
        return await self._post_json(SCHEDULE_MATCH_PATH, payload)

    async def delete_match(self, sporty_match_id: str, force: bool = False) -> dict:
        """Delete a scheduled match on the backend (also removes its live events
        and cached realtime payloads). `sporty_match_id` may be the Sporty UUID
        or the feeder external_ref. Raises on 404 (unknown); raises 409 for a
        LIVE match unless `force=True`, which overrides the guard (for cleaning
        up a simulation orphaned in `live`)."""
        suffix = "?force=true" if force else ""
        return await self._delete_json(f"{SCHEDULE_MATCH_PATH}/{sporty_match_id}{suffix}")

    async def register_players(self, payload: dict) -> dict:
        """Register the simulated lineup → returns {external_ref: sporty_player_uuid}."""
        return await self._post_json(REGISTER_PLAYERS_PATH, payload)

    async def resolve_players(self, payload: dict) -> dict:
        """Map the simulated lineup to EXISTING backend players (the ones real
        users drafted) → returns {external_ref: sporty_player_uuid} for matches."""
        return await self._post_json(RESOLVE_PLAYERS_PATH, payload)

    async def demo_setup(self, payload: dict) -> dict:
        """Ensure a demo user/league/window + fantasy lineup of the given players."""
        return await self._post_json(DEMO_SETUP_PATH, payload)

    async def push_match_result(self, payload: dict) -> bool:
        return await self._post(MATCH_RESULT_PATH, payload)

    async def push_prediction(self, payload: dict) -> bool:
        return await self._post(PREDICTION_PATH, payload)

    async def push_model_metrics(self, payload: dict) -> bool:
        """Push the /predict/metrics scorecard (model accuracy vs actual
        results) so the backend/frontend can display model performance."""
        return await self._post(MODEL_METRICS_PATH, payload)

    async def push_player_ratings(self, payload: dict) -> bool:
        return await self._post(PLAYER_RATINGS_PATH, payload)

    async def push_lineups(self, payload: dict) -> bool:
        return await self._post(MATCH_LINEUPS_PATH, payload)


_client: BackendClient | None = None


def get_backend_client() -> BackendClient:
    """Process-wide singleton so every push (simulation loop, predict,
    demo/matches orchestration) shares one connection-pooled httpx client
    instead of opening a fresh connection per call. Tests monkeypatch this
    name directly to inject a client with a mock transport, so they never
    touch the cache below. Closed from app.main's lifespan on shutdown."""
    global _client
    if _client is None:
        _client = BackendClient()
    return _client
