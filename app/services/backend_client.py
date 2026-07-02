# /home/sam069/projects/SportyDataFeeder/app/services/backend_client.py
#
# Outbound push to the Sporty backend (PRD R-4.2). httpx async, X-Feeder-Secret
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
PLAYER_RATINGS_PATH = "/api/v1/feed/player-ratings"
MATCH_LINEUPS_PATH = "/api/v1/feed/match-lineups"
SCHEDULE_MATCH_PATH = "/api/v1/feed/schedule-match"
REGISTER_PLAYERS_PATH = "/api/v1/feed/register-players"
RESOLVE_PLAYERS_PATH = "/api/v1/feed/resolve-players"
DEMO_SETUP_PATH = "/api/v1/feed/demo-setup"


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
        self._transport = transport

    async def _post(self, path: str, payload: dict) -> bool:
        url = f"{self.base_url}{path}"
        headers = {"X-Feeder-Secret": self.secret}
        last_error: str = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=REQUEST_TIMEOUT) as client:
                    response = await client.post(url, json=payload, headers=headers)
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
        async with httpx.AsyncClient(transport=self._transport, timeout=SETUP_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def _delete_json(self, path: str) -> dict:
        """DELETE and return the parsed JSON response. Single attempt; raises on
        HTTP/transport error so the caller sees 404 (unknown) / 409 (live)."""
        url = f"{self.base_url}{path}"
        headers = {"X-Feeder-Secret": self.secret}
        async with httpx.AsyncClient(transport=self._transport, timeout=SETUP_TIMEOUT) as client:
            response = await client.delete(url, headers=headers)
        response.raise_for_status()
        return response.json()

    async def schedule_match(self, payload: dict) -> dict:
        """Register a simulated fixture on the backend → returns sporty_match_id."""
        return await self._post_json(SCHEDULE_MATCH_PATH, payload)

    async def delete_match(self, sporty_match_id: str) -> dict:
        """Delete a scheduled match on the backend (also removes its live events
        and cached realtime payloads). `sporty_match_id` may be the Sporty UUID
        or the feeder external_ref. Raises on 404 (unknown) / 409 (live)."""
        return await self._delete_json(f"{SCHEDULE_MATCH_PATH}/{sporty_match_id}")

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

    async def push_player_ratings(self, payload: dict) -> bool:
        return await self._post(PLAYER_RATINGS_PATH, payload)

    async def push_lineups(self, payload: dict) -> bool:
        return await self._post(MATCH_LINEUPS_PATH, payload)


def get_backend_client() -> BackendClient:
    """Factory used by the simulation and predict endpoints; tests monkeypatch
    this to inject a client with a mock transport."""
    return BackendClient()
