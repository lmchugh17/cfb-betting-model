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
# Separate, larger threshold for confidence-tier recalibration specifically -- not the same
# question as the retrain trigger above. The 2025 holdout (934 games) already found tiers only
# weakly separated (50.3%/51.9%/53.3% cover rate); trusting a 1-2pt cover-rate gap between
# tiers on 2026 data alone needs a real sample per tier, not just 50 games. User set this
# range (150-200) 2026-09-03; using the low end so the flag fires as soon as it's meaningful.
TIER_RECAL_REVIEW_THRESHOLD = 150


def main():
    init_db()
    conn = get_connection()

    rows = conn.execute("""
        SELECT year, week, home_team, away_team, predicted_margin, actual_margin,
               win_prob_home, actual_winner, pick_team, pick_covered,
               moneyline_pick, moneyline_pick_won, margin_error, confidence_tier
        FROM prediction_results ORDER BY year, week, start_date
    """).fetchall()

    if not rows:
        print("No completed games among current predictions yet.")
        return

    # Moneyline (who wins outright) and spread (who covers) are tracked as two separate
    # picks -- often different teams, since a spread pick can be a big underdog expected
    # to lose straight-up while still covering.
    print(f"{'Matchup':<38} {'Predicted':>10} {'Actual':>8} {'Spread Pick':<16} {'ML':^4} {'ATS':^5}")
    print("-" * 90)
    ml_wins = ml_total = 0
    ats_wins = ats_losses = ats_pushes = 0
    graded_tiered = 0  # decided (non-push) spread picks that also have a confidence tier --
    # the population a tier-recalibration pass would actually segment and measure cover rate on
    errors = []
    for r in rows:
        (year, week, home, away, pred_margin, actual_margin, win_prob, actual_winner,
         pick, covered, ml_pick, ml_won, margin_error, tier) = r
        matchup = f"{away} @ {home}"
        ml_mark = "-"
        if ml_won is not None:
            ml_total += 1
            ml_wins += ml_won
            ml_mark = "W" if ml_won else "L"
        ats_mark = "-"
        if covered is not None:
            ats_mark = "W" if covered else "L"
            ats_wins += covered
            ats_losses += 1 - covered
            if tier is not None:
                graded_tiered += 1
        elif pick is not None:
            ats_mark = "P"
            ats_pushes += 1
        errors.append(margin_error)
        print(f"{matchup:<38} {pred_margin:>+9.1f} {actual_margin:>+7.0f}  "
              f"{(pick or '-'):<16} {ml_mark:^4} {ats_mark:^5}")

    print("-" * 90)
    print(f"Moneyline (straight-up): {ml_wins}-{ml_total - ml_wins} "
          f"({ml_wins/ml_total:.0%})" if ml_total else "Moneyline (straight-up): n/a")
    print(f"Against the spread: {ats_wins}-{ats_losses}-{ats_pushes}"
          + (f" ({ats_wins/(ats_wins+ats_losses):.0%})" if (ats_wins + ats_losses) else ""))
    print(f"Avg margin error: {sum(errors)/len(errors):.1f} points ({len(errors)} games)")

    if len(rows) >= RETRAIN_REVIEW_THRESHOLD:
        print(f"\n{len(rows)} completed 2026 games with reconciled predictions -- "
              f"past the {RETRAIN_REVIEW_THRESHOLD}-game review point. Worth considering a retrain "
              f"(scripts/train_model.py) that folds 2026 in as new training data.")
    else:
        print(f"\n{len(rows)}/{RETRAIN_REVIEW_THRESHOLD} completed games toward the retrain review point.")

    if graded_tiered >= TIER_RECAL_REVIEW_THRESHOLD:
        print(f"\n{graded_tiered} graded, tiered spread picks -- past the "
              f"{TIER_RECAL_REVIEW_THRESHOLD}-pick confidence-tier recalibration point. Worth "
              f"checking whether tiers separate cleanly by cover rate on 2026 data.")
    else:
        print(f"{graded_tiered}/{TIER_RECAL_REVIEW_THRESHOLD} graded, tiered spread picks "
              f"toward the confidence-tier recalibration review point.")


if __name__ == "__main__":
    main()
