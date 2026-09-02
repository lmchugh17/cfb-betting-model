"""ESPN's undocumented site API. No key required.

ESPN's own /teams list endpoint is unreliably paginated (confirmed empirically --
major FBS programs like Cincinnati and UCF are simply missing from it regardless
of limit/groups params), so team IDs are resolved individually via the search API
instead and cached in teams.espn_id.
"""
import time

import requests

SEARCH_URL = "https://site.api.espn.com/apis/search/v2"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"

# Short on purpose, same reasoning as src/weather_client.py's REQUEST_TIMEOUT_S: this is
# hit in a per-team loop (up to 138 sequential calls in scripts/scrape_injuries.py), and a
# free/unauthenticated API being called from a datacenter/CI IP range (GitHub Actions) is
# exactly the kind of traffic these APIs often throttle harder than normal residential
# requests -- confirmed for Open-Meteo 2026-09-02, not yet observed here, but the same
# structural risk applies and a short timeout costs nothing when things are working fine.
REQUEST_TIMEOUT_S = 10


def find_team_espn_id(school_name: str, mascot: str | None = None) -> str | None:
    resp = requests.get(SEARCH_URL, params={"query": school_name, "limit": 10}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()

    candidates = []
    for result_type in data.get("results", []):
        if result_type.get("type") != "team":
            continue
        for team in result_type.get("contents", []):
            if team.get("defaultLeagueSlug") != "college-football":
                continue
            uid = team.get("uid", "")  # e.g. 's:20~l:23~t:2132' -- trailing number is the team id
            if "~t:" in uid:
                candidates.append((team.get("displayName", ""), uid.split("~t:")[-1]))

    if not candidates:
        return None

    # ESPN's search ranking is unreliable for ambiguous names (e.g. "Kansas" surfaces
    # Kansas State first), so prefer an exact "{school} {mascot}" match when we have one.
    if mascot:
        exact = f"{school_name} {mascot}".lower()
        for display_name, espn_id in candidates:
            if display_name.lower() == exact:
                return espn_id

    # Next best: a candidate whose name starts with our school name as a whole word
    # (avoids "Ohio" matching "Ohio State" -- that starts with "Ohio S", not "Ohio " + boundary).
    for display_name, espn_id in candidates:
        if display_name == school_name or display_name.startswith(school_name + " "):
            if display_name.split(" ")[0:len(school_name.split(" "))] == school_name.split(" "):
                return espn_id

    return candidates[0][1]


def fetch_roster_with_injuries(espn_team_id: str) -> list:
    resp = requests.get(ROSTER_URL.format(team_id=espn_team_id), timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    players = []
    for group in data.get("athletes", []):
        players.extend(group.get("items", []))
    return players


def polite_sleep():
    time.sleep(0.3)
