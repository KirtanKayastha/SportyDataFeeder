# /home/sam069/projects/SportyDataFeeder/scripts/load_historical.py
#
# Loader for the real football-data.co.uk EPL season files in historical-datas/.
# Read-only: parses, normalizes, concatenates, and causally sorts every match.
# Produces ONE tidy DataFrame for the backtest/training harness. Does NOT touch
# the application database (the real matches stay out of the simulated `matches`
# table — see reports/OUTCOME_MODEL_TRAINING_PLAN.md, decision D3).
#
# Usage:  python -m scripts.load_historical          # prints a summary
#         from scripts.load_historical import load_matches

import sys
from pathlib import Path

import pandas as pd

HIST_DIR = Path(__file__).resolve().parents[1] / "historical-datas"

# Columns we keep when present. football-data.co.uk schema.
RESULT_COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]
STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
ODDS_COLS = ["B365H", "B365D", "B365A"]

# A few canonical team-name fixes (football-data names are mostly consistent
# across seasons, but guard the known apostrophe/spacing variants).
TEAM_CANON = {
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
}


def _season_label(d: pd.Timestamp) -> str:
    """EPL season spans Aug->May; bucket by start year."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def load_matches(hist_dir: Path | None = None) -> pd.DataFrame:
    """Return all real EPL matches as one causally sorted DataFrame.

    Columns: date, season, home, away, fthg, ftag, ftr (+ stats/odds when present).
    Rows with missing date or result are dropped. Sorted by date then a stable
    within-day order so feature construction is strictly causal.
    """
    hist_dir = Path(hist_dir) if hist_dir else HIST_DIR
    files = sorted(hist_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {hist_dir}")

    frames = []
    for f in files:
        raw = pd.read_csv(f)
        if "Date" not in raw.columns or "FTR" not in raw.columns:
            continue
        keep = ["Div", "Date", "HomeTeam", "AwayTeam"] + [
            c for c in RESULT_COLS + STAT_COLS + ODDS_COLS if c in raw.columns
        ]
        df = raw[keep].copy()
        df["date"] = pd.to_datetime(
            df["Date"], dayfirst=True, format="mixed", errors="coerce"
        )
        df = df.dropna(subset=["date", "FTHG", "FTAG", "FTR", "HomeTeam", "AwayTeam"])
        df["source_file"] = f.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["home"] = out["HomeTeam"].replace(TEAM_CANON)
    out["away"] = out["AwayTeam"].replace(TEAM_CANON)
    out["fthg"] = out["FTHG"].astype(int)
    out["ftag"] = out["FTAG"].astype(int)
    out["ftr"] = out["FTR"].astype(str)  # 'H' | 'D' | 'A'
    # Each CSV is exactly one season; label by the file's earliest date so the
    # COVID season (2019-20 ran into Jul 2020) isn't split by calendar bucketing.
    file_season = {
        fname: _season_label(grp["date"].min())
        for fname, grp in out.groupby("source_file")
    }
    out["season"] = out["source_file"].map(file_season)

    # Stable causal order: date, then home name for ties (intra-day order is
    # not market-relevant; ratings update once per day batch effectively).
    out = out.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    df = load_matches()
    seasons = df["season"].drop_duplicates().tolist()
    print(f"Loaded {len(df)} real matches across {len(seasons)} seasons")
    print(f"Divisions: {sorted(df['Div'].unique())}")
    print(f"Seasons (chronological): {seasons}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    vc = df["ftr"].value_counts(normalize=True)
    print(f"Result rate  H={vc.get('H',0):.3f}  D={vc.get('D',0):.3f}  A={vc.get('A',0):.3f}")
    print(f"Goals/match mean: {(df['fthg']+df['ftag']).mean():.2f}")
    print(f"Has odds (B365): {all(c in df.columns for c in ODDS_COLS)}")
    print(f"Per season match counts:\n{df.groupby('season').size().to_string()}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
