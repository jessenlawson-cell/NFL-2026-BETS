# 2026 NFL BETTING DATA DICTIONARY
## 1. team_metrics.csv
- `team_id`: String (2-3 letter code uppercase: e.g., KC, BUF, SF) [Primary Key]
- `season`: Integer (2026)
- `week`: Integer (Current week of data snapshot)
- `off_pass_epa`: Float (Offensive dropback Expected Points Added per play)
- `def_pass_epa`: Float (Defensive dropback EPA allowed per play)
- `off_rush_epa`: Float (Offensive rushing EPA per play)
- `def_rush_epa`: Float (Defensive rushing EPA allowed per play)
- `off_success_rate`: Float (Offensive success percentage, 0.00 to 1.00)
- `def_success_rate`: Float (Defensive success percentage allowed, 0.00 to 1.00)
- `pbwr`: Float (Pass Block Win Rate percentage, 0.00 to 1.00)
- `prwr`: Float (Pass Rush Win Rate percentage, 0.00 to 1.00)

## 2. market_odds.csv
- `game_id`: String (Format: YYYY_WW_AWAY_HOME, e.g., 2026_01_KC_BUF) [Primary Key]
- `away_team`: String (Canonical team_id for visiting team)
- `home_team`: String (Canonical team_id for home team)
- `market_spread`: Float (Point spread relative to the away team; e.g., +3.5, -2.5)
- `market_total`: Float (Consensus over/under game total; e.g., 44.5)
- `away_ml`: Integer (American moneyline odds; e.g., +150, -110)
- `home_ml`: Integer (American moneyline odds; e.g., -170, +110)

## 3. injuries.csv
- `player_id`: String (Unique identifier code) [Primary Key]
- `team_id`: String (Canonical team_id of player)
- `position`: String (Position code: QB, LT, CB, EDGE)
- `injury_status`: String (Categorical: OUT, DOUBTFUL, QUESTIONABLE)
- `snap_share_impact`: Float (Expected percentage of team snaps lost, 0.00 to 1.00)
- `replacement_quality`: Float (Performance drop-off multiplier of backup player)
