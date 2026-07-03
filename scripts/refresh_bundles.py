# /home/sam069/projects/SportyDataFeeder/scripts/refresh_bundles.py
#
# Step 4 of reports/MODEL_IMPROVEMENT_PLAN.md: keep the outcome bundles fresh.
# Elo ratings and SoT form are frozen into the pkls at build time, so a bundle
# ages as real matches are played. This one command rebuilds both production
# bundles from the latest data and (optionally) hot-swaps them into a running
# API via POST /models/reload — no restart.
#
#   python -m scripts.refresh_bundles                    # rebuild pkls only
#   python -m scripts.refresh_bundles --fetch-nba        # + re-scrape recent NBA seasons
#   python -m scripts.refresh_bundles --reload-url http://localhost:8000
#
# Football data comes from the season CSVs in historical-datas/ (drop updated
# football-data.co.uk files there); NBA recent seasons are re-fetched from
# basketball-reference with --fetch-nba (~20 polite requests).
#
# Suggested cron (weekly, Monday 04:00, with hot reload):
#   0 4 * * 1  cd /path/to/SportyDataFeeder && .venv/bin/python -m scripts.refresh_bundles --fetch-nba --reload-url http://localhost:8000 >> refresh.log 2>&1

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild production outcome bundles")
    parser.add_argument("--fetch-nba", action="store_true",
                        help="re-scrape recent NBA seasons from basketball-reference first")
    parser.add_argument("--reload-url", metavar="URL", default=None,
                        help="base URL of a running feeder API to hot-swap the "
                             "new bundles into (POST {URL}/models/reload)")
    args = parser.parse_args()

    if args.fetch_nba:
        print("== Fetching recent NBA seasons ==")
        from scripts.fetch_nba_recent import main as fetch_nba
        fetch_nba()

    print("== Rebuilding football bundle (outcome_v2.pkl) ==")
    from scripts.finalize_outcome_v2 import main as finalize_football
    finalize_football()

    print("== Rebuilding basketball bundle (outcome_v2_basketball.pkl) ==")
    from scripts.finalize_outcome_basketball import main as finalize_basketball
    finalize_basketball()

    if args.reload_url:
        import httpx

        from app.config import get_settings

        url = args.reload_url.rstrip("/") + "/models/reload"
        print(f"== Hot-swapping into {url} ==")
        response = httpx.post(
            url, headers={"X-Feeder-Secret": get_settings().FEEDER_SECRET}, timeout=30
        )
        response.raise_for_status()
        print(f"  {response.json()}")
    else:
        print("Bundles rebuilt. A running API picks them up on restart or via "
              "POST /models/reload.")


if __name__ == "__main__":
    main()
