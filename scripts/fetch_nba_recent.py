# /home/sam069/projects/SportyDataFeeder/scripts/fetch_nba_recent.py
#
# Step 2 of reports/MODEL_IMPROVEMENT_PLAN.md: the Kaggle nba.sqlite dump ends
# with the 2022-23 season, so this script fetches the missing recent seasons'
# REGULAR-SEASON results from basketball-reference.com monthly schedule pages
# and writes them to historical-datas/nba_recent_games.csv in the exact tidy
# shape produced by scripts/load_nba.py (which appends the CSV when present).
#
# Politeness: ~20 requests total, >=3.5s apart (well under basketball-reference
# rate guidance). Re-running overwrites the CSV (idempotent).
#
# Filters (applied per season, on dates — the April page has NO parseable
# "Playoffs" separator and playoff rows carry no Notes, so Notes-only filtering
# both leaks playoffs in and over-drops):
#   - everything from the first "Play-In Game"-noted date on is post-season
#     (play-in + playoffs) and is cut;
#   - of the NBA Cup / In-Season Tournament noted games, only the LAST (the
#     championship) does not count in regular-season standings — the knockout
#     rounds do and are kept;
#   - NaN-score (unplayed/cancelled) rows are dropped.
#
# Usage:  python -m scripts.fetch_nba_recent

import io
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
NBA_DB = ROOT / "nba.sqlite"
OUT_CSV = ROOT / "historical-datas" / "nba_recent_games.csv"

# Seasons missing from nba.sqlite (basketball-reference year = season END year).
SEASON_END_YEARS = [2024, 2025, 2026]
MONTHS = ["october", "november", "december", "january", "february", "march", "april"]
URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
DELAY_S = 3.5

PLAYIN_MARK = "play-in"
CUP_MARKS = ("in-season tournament", "nba cup")


def _team_map() -> dict[str, tuple[str, str]]:
    """full_name -> (franchise team_id, current abbreviation) from nba.sqlite,
    so recent games share the same Elo identity keys as the historical dump."""
    con = sqlite3.connect(NBA_DB)
    rows = con.execute("SELECT id, abbreviation, full_name FROM team").fetchall()
    con.close()
    return {name: (str(tid), abbr) for tid, abbr, name in rows}


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_month(html_text: str) -> pd.DataFrame | None:
    """Schedule table of one month page, truncated at the 'Playoffs' separator."""
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except ValueError:  # no tables on the page
        return None
    t = tables[0]
    if "Date" not in t.columns:
        return None
    # Everything from the "Playoffs" separator row on is post-season.
    sep = t.index[t["Date"].astype(str).str.strip().str.lower() == "playoffs"]
    if len(sep):
        t = t.loc[: sep[0] - 1]
    return t


def fetch_recent(end_years=SEASON_END_YEARS) -> pd.DataFrame:
    teams = _team_map()
    frames = []
    for year in end_years:
        season = f"{year - 1}-{str(year)[-2:]}"
        for month in MONTHS:
            url = URL.format(year=year, month=month)
            try:
                raw = _fetch(url)
            except Exception as exc:  # a missing month page is not fatal
                print(f"  skip {url} ({exc})")
                time.sleep(DELAY_S)
                continue
            t = _parse_month(raw)
            time.sleep(DELAY_S)
            if t is None or t.empty:
                print(f"  {season} {month}: no table")
                continue

            df = pd.DataFrame({
                "date": pd.to_datetime(t["Date"], format="%a, %b %d, %Y", errors="coerce"),
                "visitor": t["Visitor/Neutral"],
                "vpts": pd.to_numeric(t["PTS"], errors="coerce"),
                "home_name": t["Home/Neutral"],
                "hpts": pd.to_numeric(t["PTS.1"], errors="coerce"),
                "notes": t.get("Notes", pd.Series(index=t.index, dtype=object))
                          .fillna("").astype(str).str.lower(),
            }).dropna(subset=["date", "vpts", "hpts"])
            df["season"] = season
            frames.append(df)
            print(f"  {season} {month}: {len(df)} rows")

    games = pd.concat(frames, ignore_index=True)

    # Season-level regular-season filter (see header): cut from the first
    # play-in date; drop only the cup championship (last cup-noted game).
    kept = []
    for season, g in games.groupby("season"):
        playin = g[g["notes"].str.contains(PLAYIN_MARK)]
        if playin.empty:
            raise SystemExit(f"{season}: no Play-In rows found — cannot locate "
                             "the regular-season end; aborting rather than "
                             "leaking playoff games into the training data")
        g = g[g["date"] < playin["date"].min()]
        cup = g[g["notes"].str.contains("|".join(CUP_MARKS))]
        if not cup.empty:
            g = g.drop(cup[cup["date"] == cup["date"].max()].index)
        print(f"  {season}: kept {len(g)} regular-season games "
              f"(cut at {playin['date'].min().date()}, "
              f"{len(cup)} cup-noted rows)")
        kept.append(g)
    games = pd.concat(kept, ignore_index=True)
    unknown = (set(games["home_name"]) | set(games["visitor"])) - set(teams)
    if unknown:
        raise SystemExit(f"Unmapped team names (fix _team_map aliases): {unknown}")

    out = pd.DataFrame({
        "date": games["date"],
        "season": games["season"],
        "home": games["home_name"].map(lambda n: teams[n][0]),
        "away": games["visitor"].map(lambda n: teams[n][0]),
        "home_abbr": games["home_name"].map(lambda n: teams[n][1]),
        "away_abbr": games["visitor"].map(lambda n: teams[n][1]),
        "fthg": games["hpts"].astype(int),
        "ftag": games["vpts"].astype(int),
    })
    out["ftr"] = (out["fthg"] > out["ftag"]).map({True: "H", False: "A"})
    out = out.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    print(f"Fetching {len(SEASON_END_YEARS)} seasons x {len(MONTHS)} months "
          f"from basketball-reference.com ({DELAY_S}s between requests)")
    df = fetch_recent()
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} games -> {OUT_CSV}")
    print(df.groupby("season").size().to_string())
    print(f"Home win rate: {(df['ftr'] == 'H').mean():.3f}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
