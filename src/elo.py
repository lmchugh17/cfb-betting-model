"""ELO ratings for FBS teams, adapted from the FiveThirtyEight-style formula
(margin-of-victory multiplier + logistic expected score) used in the reference
NBA model, but re-calibrated for college football:

- K=40 (vs the NBA model's K=20): CFB teams play ~13 games/season vs the NBA's
  82, so each result carries far more information and ratings need to adapt
  faster within a season. This is a starting value, not empirically fit --
  worth backtesting against held-out seasons once the model exists (task 8).
- Season regression factor=0.6 (vs the NBA model's 0.75): pulls teams 40% of
  the way to the mean at each season boundary, stronger than the NBA's 25%,
  because CFB has far higher year-over-year roster turnover (graduation,
  transfer portal) than the NBA, so a team's rating should carry over less
  season-to-season.
- No ELO-diff-to-point-margin conversion is included here (the NBA reference
  model has one, but its own docstring calls it a "display-only heuristic,
  not trained" -- not real methodology). Converting elo_diff into a point
  spread or win probability is left to the actual model (task 8) to learn
  from data, rather than hand-asserting an unvalidated constant.
- Home advantage is a tunable ELO-point bonus (see HOME_ADVANTAGE_ELO below)
  applied to the home team's rating before computing expected score.
"""
from collections import defaultdict

K_FACTOR = 40
HOME_ADVANTAGE_ELO = 65  # placeholder pending calibration -- see build_features.py note
SEASON_REGRESSION_FACTOR = 0.6
INITIAL_RATING = 1500.0


class CFBElo:
    def __init__(self, k: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE_ELO):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings = defaultdict(lambda: INITIAL_RATING)
        self._current_season = None

    def get_rating(self, team) -> float:
        return self.ratings[team]

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def margin_multiplier(point_diff: float, elo_diff: float) -> float:
        mov = abs(point_diff)
        return ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))

    def maybe_regress_for_new_season(self, season):
        if self._current_season is not None and season != self._current_season:
            mean_elo = sum(self.ratings.values()) / len(self.ratings) if self.ratings else INITIAL_RATING
            for team in list(self.ratings.keys()):
                self.ratings[team] = (
                    SEASON_REGRESSION_FACTOR * self.ratings[team]
                    + (1 - SEASON_REGRESSION_FACTOR) * mean_elo
                )
        self._current_season = season

    def pre_game_features(self, home_team, away_team, neutral_site: bool) -> dict:
        """Call BEFORE update() for a game -- these are the pre-game (no-leakage) values."""
        home_bonus = 0 if neutral_site else self.home_advantage
        r_home = self.get_rating(home_team) + home_bonus
        r_away = self.get_rating(away_team)
        return {
            "elo_home": self.get_rating(home_team),
            "elo_away": self.get_rating(away_team),
            "elo_diff": self.get_rating(home_team) - self.get_rating(away_team),
            "elo_expected_home": self.expected_score(r_home, r_away),
        }

    def update(self, home_team, away_team, home_points: int, away_points: int, neutral_site: bool):
        home_bonus = 0 if neutral_site else self.home_advantage
        r_home = self.get_rating(home_team) + home_bonus
        r_away = self.get_rating(away_team)
        e_home = self.expected_score(r_home, r_away)
        s_home = 1.0 if home_points > away_points else (0.5 if home_points == away_points else 0.0)
        elo_diff = r_home - r_away
        m = self.margin_multiplier(home_points - away_points, elo_diff)
        self.ratings[home_team] = self.get_rating(home_team) + self.k * m * (s_home - e_home)
        self.ratings[away_team] = self.get_rating(away_team) + self.k * m * ((1 - s_home) - (1 - e_home))
