"""SQLite schema and connection helper for the CFB betting model."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cfb.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    school TEXT NOT NULL,
    mascot TEXT,
    abbreviation TEXT,
    espn_id TEXT
);

CREATE TABLE IF NOT EXISTS team_seasons (
    team_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    conference TEXT,
    classification TEXT,
    venue_id INTEGER,
    PRIMARY KEY (team_id, year)
);

CREATE TABLE IF NOT EXISTS venues (
    id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    state TEXT,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    capacity INTEGER,
    grass INTEGER,
    dome INTEGER
);

CREATE TABLE IF NOT EXISTS coach_seasons (
    first_name TEXT,
    last_name TEXT,
    school TEXT,
    year INTEGER,
    games INTEGER,
    wins INTEGER,
    losses INTEGER,
    ties INTEGER,
    preseason_rank INTEGER,
    postseason_rank INTEGER,
    srs REAL,
    sp_overall REAL,
    PRIMARY KEY (first_name, last_name, school, year)
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    year INTEGER,
    week INTEGER,
    season_type TEXT,
    start_date TEXT,
    neutral_site INTEGER,
    conference_game INTEGER,
    venue_id INTEGER,
    venue TEXT,
    home_id INTEGER,
    home_team TEXT,
    home_conference TEXT,
    home_points INTEGER,
    away_id INTEGER,
    away_team TEXT,
    away_conference TEXT,
    away_points INTEGER,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS game_weather (
    game_id INTEGER PRIMARY KEY,
    is_dome INTEGER NOT NULL,
    temperature_f REAL,
    wind_speed_mph REAL,
    wind_direction_deg REAL,
    precipitation_in REAL,
    humidity_pct REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS injuries (
    espn_athlete_id TEXT NOT NULL,
    scraped_at TEXT NOT NULL,
    team_id INTEGER,
    espn_team_id TEXT,
    player_name TEXT,
    position TEXT,
    status TEXT,
    injury_date TEXT,
    PRIMARY KEY (espn_athlete_id, scraped_at)
);

CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id INTEGER NOT NULL,
    team_id INTEGER,
    team TEXT NOT NULL,
    home_away TEXT,
    points INTEGER,
    first_downs INTEGER,
    total_yards INTEGER,
    net_passing_yards INTEGER,
    yards_per_pass REAL,
    completions INTEGER,
    pass_attempts INTEGER,
    passing_tds INTEGER,
    rushing_yards INTEGER,
    rushing_attempts INTEGER,
    yards_per_rush REAL,
    rushing_tds INTEGER,
    third_down_conversions INTEGER,
    third_down_attempts INTEGER,
    fourth_down_conversions INTEGER,
    fourth_down_attempts INTEGER,
    turnovers INTEGER,
    fumbles_lost INTEGER,
    total_fumbles INTEGER,
    fumbles_recovered INTEGER,
    interceptions INTEGER,
    passes_intercepted INTEGER,
    interception_yards INTEGER,
    interception_tds INTEGER,
    penalties INTEGER,
    penalty_yards INTEGER,
    possession_time_seconds INTEGER,
    sacks INTEGER,
    tackles_for_loss INTEGER,
    tackles INTEGER,
    qb_hurries INTEGER,
    passes_deflected INTEGER,
    defensive_tds INTEGER,
    kick_returns INTEGER,
    kick_return_yards INTEGER,
    kick_return_tds INTEGER,
    punt_returns INTEGER,
    punt_return_yards INTEGER,
    punt_return_tds INTEGER,
    kicking_points INTEGER,
    PRIMARY KEY (game_id, team_id)
);

CREATE TABLE IF NOT EXISTS live_odds (
    odds_game_id TEXT NOT NULL,
    scraped_at TEXT NOT NULL,
    commence_time TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome_name TEXT NOT NULL,
    price REAL,
    point REAL,
    PRIMARY KEY (odds_game_id, scraped_at, bookmaker, market, outcome_name)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    conference TEXT,
    year INTEGER NOT NULL,
    category TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    stat_value REAL,
    PRIMARY KEY (player_id, year, category, stat_type)
);

CREATE TABLE IF NOT EXISTS roster (
    player_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    first_name TEXT,
    last_name TEXT,
    team TEXT,
    position TEXT,
    jersey INTEGER,
    height INTEGER,
    weight INTEGER,
    class_year INTEGER,
    home_city TEXT,
    home_state TEXT,
    PRIMARY KEY (player_id, year)
);

CREATE TABLE IF NOT EXISTS lines (
    game_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    spread REAL,
    spread_open REAL,
    over_under REAL,
    over_under_open REAL,
    home_moneyline INTEGER,
    away_moneyline INTEGER,
    formatted_spread TEXT,
    PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS predictions (
    game_id INTEGER PRIMARY KEY,
    predicted_at TEXT NOT NULL,
    year INTEGER,
    week INTEGER,
    season_type TEXT,
    start_date TEXT,
    home_team TEXT,
    away_team TEXT,
    predicted_margin REAL,
    win_prob_home REAL,
    market_spread REAL,
    pick_team TEXT,
    edge REAL,
    confidence_tier TEXT,
    highlights_json TEXT,
    tldr TEXT,
    bullets_json TEXT,
    model_breakdown_json TEXT,
    cover_probability REAL,
    kelly_fraction REAL,
    moneyline_pick TEXT,
    moneyline_win_prob REAL,
    moneyline_confidence_tier TEXT,
    spread_price INTEGER,
    spread_price_source TEXT,
    spread_price_book_count INTEGER,
    min_current_season_games INTEGER,
    low_sample_team TEXT,
    low_sample_team_games INTEGER
);
"""

# Separate from SCHEMA and always dropped + recreated in init_db() (not IF NOT EXISTS) --
# a view's definition needs to track predictions' current columns, and a bare
# "CREATE VIEW IF NOT EXISTS" would silently keep an old view's stale column list forever
# once it exists once, exactly the bug hit adding moneyline_pick_won.
VIEW_SCHEMA = """
-- Always-live join of predictions against actual results, for both the site's
-- results section and deciding when enough new completed games have
-- accumulated to be worth a retrain (task: periodic model refresh).
CREATE VIEW prediction_results AS
SELECT
    p.*,
    g.home_points, g.away_points,
    (g.home_points - g.away_points) AS actual_margin,
    CASE WHEN g.home_points > g.away_points THEN p.home_team ELSE p.away_team END AS actual_winner,
    -- Whether the SPREAD pick also happened to win outright -- NOT the moneyline record.
    -- An ATS pick is routinely the underdog taking points, so it's expected to lose this
    -- often even when working exactly as intended. Kept for diagnostics; the site's
    -- "Straight-Up" stat uses moneyline_pick_won below instead.
    CASE WHEN p.pick_team = (CASE WHEN g.home_points > g.away_points THEN p.home_team ELSE p.away_team END)
         THEN 1 ELSE 0 END AS ats_pick_won_straight_up,
    -- The actual moneyline record: did the dedicated moneyline_pick (whichever side the
    -- model gives >50% win probability) win outright?
    CASE WHEN p.moneyline_pick IS NULL THEN NULL
         WHEN p.moneyline_pick = (CASE WHEN g.home_points > g.away_points THEN p.home_team ELSE p.away_team END)
         THEN 1 ELSE 0 END AS moneyline_pick_won,
    CASE
        WHEN p.pick_team IS NULL OR p.market_spread IS NULL THEN NULL
        WHEN (g.home_points - g.away_points) + p.market_spread = 0 THEN NULL  -- push
        WHEN p.pick_team = p.home_team THEN
            CASE WHEN (g.home_points - g.away_points) + p.market_spread > 0 THEN 1 ELSE 0 END
        WHEN p.pick_team = p.away_team THEN
            CASE WHEN (g.home_points - g.away_points) + p.market_spread < 0 THEN 1 ELSE 0 END
    END AS pick_covered,
    ABS(p.predicted_margin - (g.home_points - g.away_points)) AS margin_error
FROM predictions p
JOIN games g ON p.game_id = g.id
WHERE g.home_points IS NOT NULL AND g.away_points IS NOT NULL;
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # lets concurrent backfill scripts write without lock errors
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        existing_team_cols = {row[1] for row in conn.execute("PRAGMA table_info(teams)")}
        if "espn_id" not in existing_team_cols:
            conn.execute("ALTER TABLE teams ADD COLUMN espn_id TEXT")
        existing_pred_cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        new_pred_cols = {
            "model_breakdown_json": "TEXT", "cover_probability": "REAL", "kelly_fraction": "REAL",
            "moneyline_pick": "TEXT", "moneyline_win_prob": "REAL", "moneyline_confidence_tier": "TEXT",
            "spread_price": "INTEGER", "spread_price_source": "TEXT", "spread_price_book_count": "INTEGER",
            "min_current_season_games": "INTEGER",
            "low_sample_team": "TEXT", "low_sample_team_games": "INTEGER",
        }
        for col, sqltype in new_pred_cols.items():
            if col not in existing_pred_cols:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {sqltype}")
        # Must run after the ALTER TABLE migrations above -- the view references columns
        # (e.g. moneyline_pick) that may have just been added to an existing DB.
        conn.execute("DROP VIEW IF EXISTS prediction_results")
        conn.executescript(VIEW_SCHEMA)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
