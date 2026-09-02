"""Shared team-name matching between our own `teams` table (CFBD-sourced names) and
external sources that name schools differently -- The Odds API ("School Mascot",
e.g. "Rutgers Scarlet Knights") and Polymarket (plain school name, e.g. "UMass").
Split out so all of them share one lookup instead of drifting.
"""
import unicodedata

# CFBD and external sources occasionally use different short names for the same school.
# Purely additive -- build_team_lookup/build_school_only_lookup always keep the real
# CFBD name mapped too, an alias here just adds a second valid key alongside it.
SCHOOL_ALIASES = {
    "Southern Miss": "Southern Mississippi",
    "Sam Houston": "Sam Houston State",
    "Massachusetts": "UMass",
    "App State": "Appalachian State",
    "Miami": "Miami (FL)",  # CFBD's plain "Miami" is the FL/Hurricanes program; Polymarket
                             # disambiguates it from "Miami (OH)" the way CFBD already does for OH.
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


def build_school_only_lookup(conn) -> dict:
    """Maps normalized school name alone (no mascot) -> our team id -- Polymarket's
    event titles use plain school names ('UMass vs. Rutgers'), unlike the Odds API's
    'School Mascot' convention build_team_lookup() handles."""
    rows = conn.execute("SELECT id, school FROM teams").fetchall()
    lookup = {}
    for team_id, school in rows:
        lookup[normalize(school)] = team_id
        alias = SCHOOL_ALIASES.get(school)
        if alias:
            lookup[normalize(alias)] = team_id
    return lookup
