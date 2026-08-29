"""Reports the model's real track record: every prediction reconciled against
its actual result, via the always-live `prediction_results` view (predictions
JOIN games, computed fresh -- no separate table to keep in sync).

Also flags when enough new completed 2026 games have accumulated to be worth
folding into a retrain: the trained model only knows 2021-2025 (see
scripts/train_model.py) -- results piling up here are exactly the new season's
data a retrain would add.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db

RETRAIN_REVIEW_THRESHOLD = 50  # arbitrary first-pass number, not tuned


def main():
    init_db()
    conn = get_connection()

    rows = conn.execute("""
        SELECT year, week, home_team, away_team, predicted_margin, actual_margin,
               win_prob_home, actual_winner, pick_team, pick_won_straight_up, pick_covered,
               margin_error, confidence_tier
        FROM prediction_results ORDER BY year, week, start_date
    """).fetchall()

    if not rows:
        print("No completed games among current predictions yet.")
        return

    print(f"{'Matchup':<38} {'Predicted':>10} {'Actual':>8} {'Pick':<16} {'SU':^4} {'ATS':^5}")
    print("-" * 90)
    straight_up_wins = straight_up_total = 0
    ats_wins = ats_losses = ats_pushes = 0
    errors = []
    for r in rows:
        (year, week, home, away, pred_margin, actual_margin, win_prob, actual_winner,
         pick, won_su, covered, margin_error, tier) = r
        matchup = f"{away} @ {home}"
        su_mark = "-"
        if won_su is not None:
            straight_up_total += 1
            straight_up_wins += won_su
            su_mark = "W" if won_su else "L"
        ats_mark = "-"
        if covered is not None:
            ats_mark = "W" if covered else "L"
            ats_wins += covered
            ats_losses += 1 - covered
        elif pick is not None:
            ats_mark = "P"
            ats_pushes += 1
        errors.append(margin_error)
        print(f"{matchup:<38} {pred_margin:>+9.1f} {actual_margin:>+7.0f}  "
              f"{(pick or '-'):<16} {su_mark:^4} {ats_mark:^5}")

    print("-" * 90)
    print(f"Straight-up: {straight_up_wins}-{straight_up_total - straight_up_wins} "
          f"({straight_up_wins/straight_up_total:.0%})" if straight_up_total else "Straight-up: n/a")
    print(f"Against the spread: {ats_wins}-{ats_losses}-{ats_pushes}"
          + (f" ({ats_wins/(ats_wins+ats_losses):.0%})" if (ats_wins + ats_losses) else ""))
    print(f"Avg margin error: {sum(errors)/len(errors):.1f} points ({len(errors)} games)")

    if len(rows) >= RETRAIN_REVIEW_THRESHOLD:
        print(f"\n{len(rows)} completed 2026 games with reconciled predictions -- "
              f"past the {RETRAIN_REVIEW_THRESHOLD}-game review point. Worth considering a retrain "
              f"(scripts/train_model.py) that folds 2026 in as new training data.")
    else:
        print(f"\n{len(rows)}/{RETRAIN_REVIEW_THRESHOLD} completed games toward the retrain review point.")


if __name__ == "__main__":
    main()
