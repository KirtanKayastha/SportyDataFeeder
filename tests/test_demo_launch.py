# Tests for the one-call demo launcher (/demo/launch). The Sporty backend is
# mocked, so this verifies the feeder-side orchestration: schedule → register →
# link → simulate, and that pushed events then carry the linked sporty ids.

import app.routers.demo as demo_router


class FakeBackend:
    """Stand-in for the Sporty backend feed API."""

    MATCH_UUID = "11111111-1111-1111-1111-111111111111"

    def __init__(self, resolvable=None):
        self.scheduled = []
        self.registered = []
        self.resolved = []
        self.demo_setups = []
        self.match_results = []
        # resolve_existing only matches players whose name is in this set.
        self.resolvable = resolvable

    async def schedule_match(self, payload):
        self.scheduled.append(payload)
        return {"sporty_match_id": self.MATCH_UUID, "external_ref": "ref", "created": True}

    async def register_players(self, payload):
        self.registered.append(payload)
        players = {
            entry["external_ref"]: f"22222222-0000-0000-0000-{i:012d}"
            for i, entry in enumerate(payload["players"])
        }
        return {"status": "ok", "players": players, "created": len(players)}

    async def resolve_players(self, payload):
        self.resolved.append(payload)
        players, details = {}, []
        for i, entry in enumerate(payload["players"]):
            if self.resolvable is not None and entry["name"] not in self.resolvable:
                continue  # an existing player the user drafted; unmatched ones skip
            uuid = f"33333333-0000-0000-0000-{i:012d}"
            players[entry["external_ref"]] = uuid
            details.append({"external_ref": entry["external_ref"], "sporty_player_id": uuid,
                            "name": entry["name"], "real_team": entry.get("real_team"),
                            "external_api_id": None})
        return {"status": "ok", "players": players, "details": details, "matched": len(players)}

    async def demo_setup(self, payload):
        self.demo_setups.append(payload)
        return {"status": "ok", "fantasy_team_id": "team", "lineup_size": len(payload["player_uuids"])}

    async def push_match_result(self, payload):
        self.match_results.append(payload)
        return True

    async def push_player_ratings(self, payload):
        return True

    async def push_prediction(self, payload):
        return True


def _make_world(client):
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Linkpool", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Linksenal", "sport_id": sport_id}).json()["id"]
    for team_id, prefix in ((home, "H"), (away, "A")):
        for i in range(3):
            client.post("/players", json={
                "name": f"{prefix} Player {i}", "team_id": team_id, "position": "M", "sport_id": sport_id,
            })
    return client.post("/matches", json={
        "home_team_id": home, "away_team_id": away,
        "match_date": "2026-06-30T18:00:00", "sport_id": sport_id,
    }).json()["id"]


def test_demo_launch_schedules_links_and_simulates(client, wait_for_simulation, monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(demo_router, "get_backend_client", lambda: fake)

    match_id = _make_world(client)
    resp = client.post("/demo/launch", json={"match_id": match_id})
    assert resp.status_code == 200
    body = resp.json()

    assert body["sporty_match_id"] == FakeBackend.MATCH_UUID
    assert body["players_linked"] == 6          # 3 + 3
    assert fake.scheduled and fake.registered    # backend was called
    assert fake.demo_setups                       # fantasy demo setup invoked
    assert len(fake.registered[0]["players"]) == 6

    # The feeder created match + player entity links.
    links = client.get("/links").json()
    entities = {link["feeder_entity"] for link in links}
    assert "match" in entities and "player" in entities

    # Simulation runs and pushes match results carrying the linked sporty match id.
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"
    assert fake.match_results
    assert fake.match_results[0]["sporty_match_id"] == FakeBackend.MATCH_UUID


def test_demo_prepare_mode_registers_without_simulating(client, monkeypatch):
    """Real-league flow: prepare (schedule + register + link) without simulating
    or creating a throwaway demo team, so users can draft the registered players."""
    fake = FakeBackend()
    monkeypatch.setattr(demo_router, "get_backend_client", lambda: fake)

    match_id = _make_world(client)
    resp = client.post("/demo/launch", json={
        "match_id": match_id, "simulate": False, "fantasy_demo": False,
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "prepared"
    assert body["status_url"] is None             # not simulating yet
    assert len(body["draftable_players"]) == 6     # exposed for drafting
    assert fake.scheduled and fake.registered
    assert not fake.demo_setups                    # no throwaway team
    assert not fake.match_results                  # simulation did not run

    # Players are linked, ready to be drafted in the backend then simulated.
    links = client.get("/links").json()
    assert {l["feeder_entity"] for l in links} >= {"match", "player"}


def test_demo_resolve_existing_links_only_matched_players(client, monkeypatch):
    """Real-league flow: map the simulated lineup onto EXISTING backend players
    (the ones users drafted) by name, creating no new players. Only matched
    players are linked, so only they will score."""
    # Only two of the six simulated players exist in the backend.
    fake = FakeBackend(resolvable={"H Player 0", "A Player 1"})
    monkeypatch.setattr(demo_router, "get_backend_client", lambda: fake)

    match_id = _make_world(client)
    resp = client.post("/demo/launch", json={
        "match_id": match_id, "simulate": False, "fantasy_demo": False,
        "resolve_existing": True,
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["mode"] == "resolve_existing"
    assert body["players_in_lineup"] == 6
    assert body["players_linked"] == 2          # only the existing/drafted players
    assert fake.resolved and not fake.registered  # resolve, never create
    names = {p["name"] for p in body["draftable_players"]}
    assert names == {"H Player 0", "A Player 1"}


def test_demo_featured_player_is_simulated_first(client, monkeypatch):
    """featured_players forces a named (drafted) player into the lineup ahead of
    the default lowest-id 11, so the simulated match credits that player."""
    fake = FakeBackend()
    monkeypatch.setattr(demo_router, "get_backend_client", lambda: fake)

    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Palace", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Leeds", "sport_id": sport_id}).json()["id"]
    # 14 home players so the featured one (added last → highest id) is outside the
    # default first-11 unless featuring pulls it in.
    for i in range(14):
        client.post("/players", json={"name": f"Home Filler {i}", "team_id": home,
                                      "position": "MID", "sport_id": sport_id})
    client.post("/players", json={"name": "Jørgen Strand Larsen", "team_id": home,
                                  "position": "FWD", "sport_id": sport_id})
    for i in range(11):
        client.post("/players", json={"name": f"Away {i}", "team_id": away,
                                      "position": "MID", "sport_id": sport_id})
    match_id = client.post("/matches", json={
        "home_team_id": home, "away_team_id": away,
        "match_date": "2026-06-30T18:00:00", "sport_id": sport_id,
    }).json()["id"]

    resp = client.post("/demo/launch", json={
        "match_id": match_id, "simulate": False, "fantasy_demo": False,
        "featured_players": ["Strand Larsen"],
    })
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()["draftable_players"]]
    assert "Jørgen Strand Larsen" in names  # pulled into the lineup despite high id


def test_demo_launch_unknown_match_404(client):
    assert client.post("/demo/launch", json={"match_id": 99999}).status_code == 404
