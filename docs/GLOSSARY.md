# Glossary

Every abbreviation and domain term used across `docs/*.md`, defined once here. Terms are
grouped by topic; within a group they are alphabetical.

## General software terms

| Term | Full form / meaning |
|---|---|
| API | Application Programming Interface |
| ASGI | Asynchronous Server Gateway Interface — the async successor to WSGI; FastAPI + Uvicorn run on it |
| CI/CD | Continuous Integration / Continuous Delivery |
| CLI | Command-Line Interface |
| CORS | Cross-Origin Resource Sharing |
| CRUD | Create, Read, Update, Delete |
| CSRF | Cross-Site Request Forgery |
| CSV | Comma-Separated Values |
| DB | Database |
| DI | Dependency Injection |
| FK | Foreign Key |
| HTTP | Hypertext Transfer Protocol |
| HTTPS | HTTP Secure |
| JSON | JavaScript Object Notation |
| JWT | JSON Web Token (**not used in this codebase** — see `CONFIGURATION_DEPLOYMENT.md`) |
| ORM | Object-Relational Mapper (SQLAlchemy, in this codebase) |
| PK | Primary Key |
| SDK | Software Development Kit |
| SQL | Structured Query Language |
| TTL | Time To Live |
| UI | User Interface |
| UUID | Universally Unique Identifier |
| WS | WebSocket |
| XSS | Cross-Site Scripting |

## ML / statistics terms

| Term | Full form / meaning |
|---|---|
| ECE | Expected Calibration Error — a metric measuring how well a model's stated confidence matches its actual accuracy; computed in `app/services/prediction_metrics.py` and `scripts/backtest_outcome.py` |
| Elo / Elo rating | A relative skill-rating system (named after its creator, Arpad Elo) originally built for chess; here it rates football/basketball teams. See `app/services/team_ratings.py` |
| EWMA | Exponentially Weighted Moving Average — used for player "form" in `app/services/features.py` |
| FTE Elo / FiveThirtyEight multiplier | A margin-of-victory scaling formula for Elo updates popularized by the FiveThirtyEight website's NBA/NFL Elo models; implemented as `EloModel._mov_multiplier(mov="fte")` |
| GBM | Gradient-Boosted Machine (mentioned only as a rejected alternative in `reports/MODEL_IMPROVEMENT_PLAN.md` — **not used** in this codebase) |
| L2 (regularisation) | Ridge-style penalty term (sum of squared parameters) added to a loss function to discourage large parameter values; used in `app/services/dixon_coles.py` |
| lbfgs | Limited-memory Broyden–Fletcher–Goldfarb–Shanno — a quasi-Newton numerical optimisation algorithm; the default/only solver used for `LogisticRegression` in this codebase |
| Log loss (logarithmic loss / cross-entropy loss) | A probabilistic scoring rule that heavily penalizes confident-and-wrong predictions; the primary model-selection metric throughout `scripts/backtest_*.py` |
| MAE | Mean Absolute Error (mentioned in the doc-generation brief; **not computed anywhere in this codebase** — the codebase's regression-style error metric is Brier score, not MAE) |
| ML | Machine Learning |
| MOV | Margin Of Victory — an Elo-update weighting scheme that scales the rating change by how large the winning margin was, not just win/loss |
| MSE | Mean Squared Error (**not used** in this codebase; see Brier score below, which is MSE applied to a one-hot outcome vector) |
| NLL | Negative Log-Likelihood — the quantity `scipy.optimize.minimize` minimizes when fitting the Dixon-Coles model |
| OOS | Out-Of-Sample — predictions scored on data the model was not trained on, i.e. genuine test performance |
| PMF | Probability Mass Function — used for the Poisson goal-count distribution in `app/services/dixon_coles.py` |
| RMSE | Root Mean Squared Error (**not used** in this codebase) |
| ROC / AUC | Receiver Operating Characteristic / Area Under the Curve (**not used** in this codebase — the code is multi-class and uses log loss/Brier/ECE instead) |
| SoT | Shots on Target — a football statistic used as a rolling team-form feature in `app/services/features_team.py` |
| xG | Expected Goals — a well-known football analytics concept describing shot quality; **not implemented in this codebase** (no `xg` code exists; shots-on-target form is used as a lower-effort proxy instead). Listed here only because the documentation brief asked about it explicitly. |

## Domain / project-specific terms

| Term | Meaning |
|---|---|
| Backend / "the Sporty backend" | The separate main Sporty application (`~/projects/Sporty/Sporty_Backend`) that owns fantasy-league logic, user accounts, and the real-time WebSocket fan-out. Outside the scope of this repository except as an HTTP client target. |
| Bundle | A pickled Python dict saved to `models_pkl/*.pkl` holding a trained model plus everything needed to use it at inference time (feature list, Elo ratings, aliases, etc.) — see `MODELS.md` |
| Cold start | The state of a player/team with no (or insufficient) historical stat rows; the system falls back to league-average constants rather than raising an error |
| Entity link | A row in the `entity_links` table mapping one of the feeder's own integer IDs (for a `sport`/`team`/`player`/`match`) to the Sporty backend's string UUID for the same real-world entity |
| Feeder | This codebase / service — "Sporty Data Feeder" |
| Feeder-owned ID | An auto-increment integer primary key from the feeder's own Postgres tables, as opposed to a Sporty backend UUID |
| Fixture | A scheduled match (a row in the `matches` table before/without a final result) |
| Gameweek | A round-numbered slice of a football season used for grouping per-player stat rows (`player_stats.gameweek`); `0` means "season-to-date" totals rather than a single round |
| Lineup | The set of players simulated as "on the pitch/court" for a match — always exactly 11 (football) or 5 (basketball) players, selected deterministically by lowest player id (or by name match for "featured" players) |
| Man of the match / MOTM | The single highest-rated player in a finished simulation, computed by `app/services/rater.py:find_man_of_match` |
| Pkl / pickle | Python's native binary object-serialization format (the `pickle` module); used to persist trained models to disk |
| Push | An outbound authenticated HTTP POST from the feeder to the Sporty backend's `/api/v1/feed/*` endpoints |
| Replay / replay-push | Re-sending all of a match's already-stored events to the Sporty backend, used for outage recovery; safe because the backend deduplicates on `event_id` |
| Sport type | One of the `SportType` enum values (`FOOTBALL`, `BASKETBALL`, `CRICKET`, `UNKNOWN`) resolved from a free-form sport name string by `app/services/sport_resolver.py` |
| Stint | In the basketball rotation model, the number of consecutive minutes a player has been on the court without a rest |
| Walk-forward validation / expanding-window backtest | A time-series-safe evaluation method used throughout `scripts/backtest_*.py` and `scripts/train_outcome_v*.py`: train only on seasons strictly before a test season, evaluate on that season, then expand the training window and repeat for the next season |
