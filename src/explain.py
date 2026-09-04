"""Turns a game's model prediction + feature values into the grounded facts
that drive a pick explanation.

Only the deterministic half lives here: extract the actual top-contributing
features for THIS specific game via XGBoost's native SHAP (pred_contribs) on
the margin regressor, then render each into a factual sentence using the
game's real numbers. No LLM call in this module -- these facts are true by
construction.

The prose-writing half (turning these facts into a TL;DR + bullets) is done
natively by the Claude Code session that runs the weekly pipeline, not by a
separate Anthropic API call -- there's no need for this project to hold its
own billed API key when the automation already runs inside a Claude session
with its own access. See scripts/weekly_pipeline.py (task 11) for how these
highlights get used.
"""
import pandas as pd
import xgboost as xgb

from src.weather_features import ADVERSE_PRECIP_IN, ADVERSE_TEMP_F, ADVERSE_WIND_MPH

TOP_N_FACTORS = 5


def get_shap_contributions(regressor, X_row: pd.DataFrame, feature_columns: list) -> dict:
    """Per-game SHAP contributions toward the predicted margin, via XGBoost's
    built-in TreeSHAP (no separate `shap` package needed)."""
    dmat = xgb.DMatrix(X_row[feature_columns])
    contribs = regressor.get_booster().predict(dmat, pred_contribs=True)[0]
    return dict(zip(feature_columns, contribs[:-1]))  # last column is the bias term


def _describe_weather_condition(row: dict) -> str:
    """Names whichever adverse-weather threshold(s) this game's forecast actually
    crossed -- thresholds match src.weather_features so the description never drifts
    from what the feature itself was gated on."""
    temp, wind = row.get("weather_temperature_f"), row.get("weather_wind_speed_mph")
    precip = row.get("weather_precipitation_in")
    parts = []
    if temp is not None and temp <= ADVERSE_TEMP_F:
        parts.append(f"{temp:.0f}°F temperatures")
    if wind is not None and wind >= ADVERSE_WIND_MPH:
        parts.append(f"{wind:.0f} mph wind")
    if precip is not None and precip >= ADVERSE_PRECIP_IN:
        parts.append(f"{precip:.2f}in of precipitation")
    return " and ".join(parts) if parts else "adverse weather"


def _describe_feature(feature: str, row: dict, home_team: str, away_team: str) -> str | None:
    """Renders one feature into a grounded, factual sentence using this game's real values."""
    g = lambda k, default=None: row.get(k, default)

    if feature == "elo_diff":
        diff = g("elo_diff")
        leader, trailer = (home_team, away_team) if diff > 0 else (away_team, home_team)
        # &Dagger; matches the ELO footnote marker used in scripts/build_site.py's methodology
        # paragraph and footer definition -- attached here too so a reader landing on any
        # individual game card (not just the footer) can trace "ELO" back to its explanation.
        return f"{leader} carries a {abs(diff):.0f}-point ELO&Dagger; advantage over {trailer} ({g('elo_home'):.0f} vs {g('elo_away'):.0f})."
    if feature == "elo_expected_home":
        return f"Pre-game ELO&Dagger; gives {home_team} a {g('elo_expected_home'):.0%} win probability."
    if feature == "srs_diff":
        diff = g("srs_diff")
        leader, trailer = (home_team, away_team) if diff > 0 else (away_team, home_team)
        return (f"{leader} has been the better team once schedule strength is accounted for, "
                f"outrating {trailer} by {abs(diff):.1f} points of opponent-adjusted margin "
                f"({g('srs_home'):+.1f} vs {g('srs_away'):+.1f}).")
    if feature in ("home_ats_pct", "away_ats_pct"):
        team = home_team if feature == "home_ats_pct" else away_team
        pct = g(feature)
        n = g("home_ats_count" if feature == "home_ats_pct" else "away_ats_count")
        # pct is nan (not None) for a team with zero decided ATS games -- pd.isna catches
        # NaN, unlike the "is not None" check this replaced, which let "nan%" print verbatim.
        if not pd.notna(pct):
            return None
        return f"{team} has covered the spread in {pct:.0%} of its last {int(n)} game{'s' if n != 1 else ''}."
    if feature in ("home_rest_days", "away_rest_days"):
        team = home_team if feature == "home_rest_days" else away_team
        days = g(feature)
        bye = g("home_bye_week" if feature == "home_rest_days" else "away_bye_week")
        return f"{team} enters off a bye week ({days:.0f} days rest)." if bye else f"{team} has {days:.0f} days of rest."
    if feature == "h2h_avg_home_margin" and g("h2h_meetings", 0):
        margin = g("h2h_avg_home_margin")
        favored, other = (home_team, away_team) if margin > 0 else (away_team, home_team)
        return (f"Over their last {int(g('h2h_meetings'))} meetings, {favored} has outscored {other} "
                f"by an average of {abs(margin):.1f} points.")
    if feature == "is_adverse_weather":
        if not g("is_adverse_weather"):
            return None
        return f"This game is forecast for {_describe_weather_condition(row)}."
    if feature == "adverse_wx_ats_edge":
        edge = g("adverse_wx_ats_edge")
        if not g("is_adverse_weather") or not edge:
            return None  # not itself an adverse-weather game, or a genuine tie/missing-history zero
        home_pct, away_pct = g("home_adverse_wx_ats_pct"), g("away_adverse_wx_ats_pct")
        home_n, away_n = g("home_adverse_wx_ats_count"), g("away_adverse_wx_ats_count")
        leader, trailer = (home_team, away_team) if edge > 0 else (away_team, home_team)
        leader_pct, trailer_pct = (home_pct, away_pct) if edge > 0 else (away_pct, home_pct)
        leader_n, trailer_n = (home_n, away_n) if edge > 0 else (away_n, home_n)
        return (f"In its last {int(leader_n)} game{'s' if leader_n != 1 else ''} with "
                f"{_describe_weather_condition(row)}, {leader} has covered the spread {leader_pct:.0%} "
                f"of the time, versus {trailer}'s {trailer_pct:.0%} over its last {int(trailer_n)} "
                f"such game{'s' if trailer_n != 1 else ''}.")
    if feature.startswith("diff_avg_"):
        stat = feature.replace("diff_avg_", "")
        home_val, away_val, diff = g(f"home_avg_{stat}"), g(f"away_avg_{stat}"), g(feature)
        if home_val is None or away_val is None or pd.isna(diff):
            return None
        leader, trailer = (home_team, away_team) if diff > 0 else (away_team, home_team)
        if stat == "third_down_pct":
            return f"{leader} has been converting third downs at a higher rate over their last 4 games ({home_val:.0%} vs {away_val:.0%})."
        if stat == "opponent_srs":
            tougher, easier = (home_team, away_team) if diff > 0 else (away_team, home_team)
            tougher_val, easier_val = max(home_val, away_val), min(home_val, away_val)
            return f"{tougher} has faced tougher recent competition than {easier} (average opponent rating {tougher_val:+.1f} vs {easier_val:+.1f})."
        readable_stat = {
            "total_yards": "total yards per game", "rushing_yards": "rushing yards per game",
            "net_passing_yards": "passing yards per game", "yards_per_play": "yards per play",
            "turnover_margin": "turnover margin", "point_diff": "scoring margin", "win": "win rate",
        }.get(stat, stat)
        return f"{leader} has the edge in recent {readable_stat} ({home_val:.1f} vs {away_val:.1f}, last 4 games)."
    return None


def build_feature_highlights(row: dict, contributions: dict, home_team: str, away_team: str,
                              top_n: int = TOP_N_FACTORS) -> list[str]:
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    highlights = []
    for feature, _ in ranked:
        sentence = _describe_feature(feature, row, home_team, away_team)
        if sentence:
            highlights.append(sentence)
        if len(highlights) >= top_n:
            break
    return highlights
