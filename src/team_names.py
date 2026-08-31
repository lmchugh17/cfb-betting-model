"""Shared team-name matching between our own `teams` table (CFBD-sourced names) and
external sources that name schools differently -- currently just The Odds API
("School Mascot", e.g. "Rutgers Scarlet Knights"), used by scripts/pull_odds.py and
src/spread_pricing.py. Split out so both can share one lookup instead of drifting.
"""
import unicodedata

# CFBD and The Odds API occasionally use different short names for the same school.
SCHOOL_ALIASES = {
    "Southern Miss": "Southern Mississippi",
    "Sam Houston": "Sam Houston State",
    "Massachusetts": "UMass",
    "App State": "Appalachian State",
}


def normalize(name: str) -> str:
    """Strips diacritics (Hawai'i/San José -> Hawaii/San Jose) and apostrophes for matching."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.replace("'", "").replace("’", "")


def build_team_lookup(conn) -> dict:
    """Maps normalized 'School Mascot' (as the Odds API names teams) -> our team id."""
    rows = conn.execute("SELECT id, school, mascot FROM teams").fetchall()
    lookup = {}
    for team_id, school, mascot in rows:
        lookup[normalize(f"{school} {mascot}")] = team_id
        alias = SCHOOL_ALIASES.get(school)
        if alias:
            lookup[normalize(f"{alias} {mascot}")] = team_id
    return lookup
