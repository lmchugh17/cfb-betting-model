"""Archives and prunes the two time-series snapshot tables (live_odds, injuries)
that grow unboundedly -- both key their primary key on scraped_at, so every pull
writes a brand-new snapshot instead of updating existing rows. Measured 2026-08-29:
one live_odds pull alone wrote 3,038 rows (~864KB); at the season's Tue/Fri/Sat
cadence that projects to ~45-50MB of pure accumulation by season's end, pushing
data/cfb.db toward GitHub's 100MB push limit. Run this after each pull
(pull_odds.py, scrape_injuries.py) to keep it bounded.

Neither table is currently read by training (scripts/build_features.py) or live
inference (scripts/predict_games.py) -- both get market_spread from the `lines`
table instead -- so pruning is safe for the model today. But live_odds is the only
source with real per-book spread pricing (`lines` only has the point, not the
price -- see the ASSUMED_SPREAD_ODDS_AMERICAN simplification in predict_games.py)
and a genuine line-movement time series, both plausible future features. Rather
than delete that history outright, stale rows are archived to an append-only CSV
first (data/live_odds_archive.csv, data/injuries_archive.csv -- explicitly
un-ignored in .gitignore and committed, since GitHub Actions runners have no
persistent disk between runs) and only then removed from the SQLite database. This
also directly addresses the git-history-bloat half of the growth problem: appending
new text lines to a CSV compresses far better in git's history than re-committing a
fully rewritten multi-MB binary SQLite blob on every pull.

Usage: .venv/bin/python scripts/prune_live_data.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# live_odds is keyed on scraped_at, so it accumulates a full new snapshot per
# game/bookmaker/market/outcome on every pull. Once a game's kicked off there's no
# more line-shopping value in it -- a few days of buffer covers any lag between a
# game finishing and this script's next run.
LIVE_ODDS_RETENTION_DAYS = 3

# injuries has no natural "game" tie -- it's a roster-wide snapshot, not keyed to a
# specific matchup. Only the most recent status per player matters for the ~2-week
# window predict_games.py actually looks at, so a longer buffer than live_odds is fine.
INJURIES_RETENTION_DAYS = 14


def _archive_and_prune(conn, table: str, time_col: str, cutoff_iso: str, archive_path: Path) -> int:
    stale = pd.read_sql(f"SELECT * FROM {table} WHERE {time_col} IS NOT NULL AND {time_col} < ?",
                         conn, params=(cutoff_iso,))
    if stale.empty:
        return 0
    write_header = not archive_path.exists()
    stale.to_csv(archive_path, mode="a", header=write_header, index=False)
    conn.execute(f"DELETE FROM {table} WHERE {time_col} IS NOT NULL AND {time_col} < ?", (cutoff_iso,))
    conn.commit()
    return len(stale)


def main():
    init_db()
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc)

        odds_cutoff = (now - timedelta(days=LIVE_ODDS_RETENTION_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        n_odds = _archive_and_prune(conn, "live_odds", "commence_time", odds_cutoff,
                                     DATA_DIR / "live_odds_archive.csv")
        print(f"live_odds: archived + pruned {n_odds} row(s) for games before {odds_cutoff}")

        inj_cutoff = (now - timedelta(days=INJURIES_RETENTION_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        n_inj = _archive_and_prune(conn, "injuries", "scraped_at", inj_cutoff,
                                    DATA_DIR / "injuries_archive.csv")
        print(f"injuries: archived + pruned {n_inj} row(s) scraped before {inj_cutoff}")

        if n_odds or n_inj:
            conn.execute("VACUUM")
            print("Vacuumed database to reclaim freed space.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
