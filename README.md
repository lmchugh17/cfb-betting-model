# CFB Betting Model

**Live site: [lmchugh17.github.io/cfb-betting-model](https://lmchugh17.github.io/cfb-betting-model/)**

Opponent-adjusted college football predictions vs. the market. A 5-model stacked
ensemble (logistic regression, random forest, XGBoost, LightGBM, extra trees) for
win probability, plus a separate XGBoost regressor for predicted margin, trained on
2021-2024 and evaluated on a full held-out 2025 season. Personal research project,
not financial advice.

## How it runs

- **Data pull** (`.github/workflows/weekly_data_pull.yml`, GitHub Actions): games,
  lines, player stats, live odds, injuries, and weather forecasts, Tue/Fri/Sat mornings.
- **Predictions** (a scheduled Claude Code routine, ~1 hour after the data pull): picks
  upcoming games, writes grounded explanations from the model's own SHAP-ranked
  factors, and rebuilds the site.
- `docs/index.html` is the generated static site GitHub Pages serves from `/docs` on
  `master` — it's a build output, not something to hand-edit. Regenerate it via
  `scripts/build_site.py`.

See the site's own footer for full methodology, confidence-tier definitions, and the
paper-bankroll/Kelly-sizing assumptions.
