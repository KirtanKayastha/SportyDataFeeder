# /home/sam069/projects/SportyDataFeeder/scripts/fetch_championship.py
#
# Step 5 of reports/MODEL_IMPROVEMENT_PLAN.md (multi-league pooling): download
# the English Championship (football-data.co.uk division E1) season files that
# parallel the EPL seasons in historical-datas/. Pooling the second tier into
# the Elo pass means promoted teams arrive in the EPL with a real rating
# (earned against previously-relegated sides) instead of the 1500 cold start,
# and relegated teams keep theirs.
#
# Files land in historical-datas/championship/ (a subdirectory, so the default
# EPL-only loader glob is unaffected). Idempotent: re-running overwrites.
#
# Usage:  python -m scripts.fetch_championship

import sys
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "historical-datas" / "championship"
URL = "https://www.football-data.co.uk/mmz4281/{code}/E1.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64)"
# Same seasons as the EPL files (2013-14 .. 2025-26).
SEASON_CODES = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2013, 2026)]
DELAY_S = 1.5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code in SEASON_CODES:
        url = URL.format(code=code)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        out = OUT_DIR / f"E1_{code}.csv"
        out.write_bytes(data)
        print(f"  {out.name}: {len(data)} bytes")
        time.sleep(DELAY_S)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
