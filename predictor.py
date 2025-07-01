import pandas as pd
import difflib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Load the dataset
df = pd.read_excel("2024-2025.xlsx")

# Get all unique player names
player_names = list(set(df["Player_1"].dropna().tolist() + df["Player_2"].dropna().tolist()))

def calculate_elo_ratings(df, k=32, base_elo=1500):
    elo_dict = {}
    elo_p1_pre = []
    elo_p2_pre = []
    elo_win_prob_p1 = []
    for idx, row in df.sort_values('Date').iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        r1 = elo_dict.get(p1, base_elo)
        r2 = elo_dict.get(p2, base_elo)
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1
        elo_p1_pre.append(r1)
        elo_p2_pre.append(r2)
        elo_win_prob_p1.append(e1)
        s1 = 1 if row['Winner'] == p1 else 0
        s2 = 1 - s1
        r1_new = r1 + k * (s1 - e1)
        r2_new = r2 + k * (s2 - e2)
        elo_dict[p1] = r1_new
        elo_dict[p2] = r2_new
    df['elo_p1_pre'] = elo_p1_pre
    df['elo_p2_pre'] = elo_p2_pre
    df['elo_diff'] = df['elo_p1_pre'] - df['elo_p2_pre']
    df['elo_win_prob_p1'] = elo_win_prob_p1
    return df

def calculate_h2h_features(df, player1, player2, up_to_date=None):
    h2h = df[
        (((df["Player_1"] == player1) & (df["Player_2"] == player2)) |
         ((df["Player_1"] == player2) & (df["Player_2"] == player1)))
    ]
    if up_to_date is not None:
        h2h = h2h[h2h["Date"] < up_to_date]
    h2h = h2h.sort_values("Date", ascending=True)
    h2h_matches = len(h2h)
    if h2h_matches == 0:
        return [0, 0, 0, 0, 0]
    h2h_wins_p1 = len(h2h[h2h["Winner"] == player1])
    h2h_winrate_p1 = h2h_wins_p1 / h2h_matches
    h2h_recent_p1_win = int(h2h.iloc[-1]["Winner"] == player1)
    streak = 0
    for _, row in h2h[::-1].iterrows():
        if row["Winner"] == player1:
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break
    h2h_streak_p1 = streak
    return [h2h_matches, h2h_wins_p1, h2h_winrate_p1, h2h_recent_p1_win, h2h_streak_p1]

# Preprocessing
df.dropna(subset=["Winner", "Player_1", "Player_2"], inplace=True)
df["Surface"] = df["Surface"].str.lower()
df["Court"] = df["Court"].str.lower()
df["Round"] = df["Round"].str.lower()
df["Series"] = df["Series"].str.lower()
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y", errors='coerce')
df = df.dropna(subset=["Date"])

def create_features(df):
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["odds_diff"] = df["Odd_1"] - df["Odd_2"]
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
    df["odds_ratio"] = df["Odd_1"] / df["Odd_2"]
    df["implied_prob_1"] = 1 / df["Odd_1"]
    df["implied_prob_2"] = 1 / df["Odd_2"]
    df["prob_diff"] = df["implied_prob_1"] - df["implied_prob_2"]
    df["rank_ratio"] = df["Rank_1"] / df["Rank_2"]
    df["target"] = (df["Winner"] == df["Player_1"]).astype(int)
    return df

def calculate_player_form(df, player_name, surface=None, recent_matches=10):
    player_matches = df[(df["Player_1"] == player_name) | (df["Player_2"] == player_name)]
    if surface:
        player_matches = player_matches[player_matches["Surface"] == surface]
    if player_matches.empty:
        return 0.5
    recent = player_matches.sort_values("Date", ascending=False).head(recent_matches)
    wins = len(recent[recent["Winner"] == player_name])
    return wins / len(recent) if len(recent) > 0 else 0.5

def calculate_surface_stats(df, player_name, surface):
    surface_matches = df[
        ((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) &
        (df["Surface"] == surface)
    ]
    if surface_matches.empty:
        return 0.5
    wins = len(surface_matches[surface_matches["Winner"] == player_name])
    total = len(surface_matches)
    return wins / total

def add_player_stats(df):
    df["player1_form"] = 0.5
    df["player2_form"] = 0.5
    df["player1_surface_winrate"] = 0.5
    df["player2_surface_winrate"] = 0.5
    for idx, row in df.iterrows():
        temp_df = df[df["Date"] < row["Date"]]
        df.at[idx, "player1_form"] = calculate_player_form(temp_df, row["Player_1"], row["Surface"])
        df.at[idx, "player2_form"] = calculate_player_form(temp_df, row["Player_2"], row["Surface"])
        df.at[idx, "player1_surface_winrate"] = calculate_surface_stats(temp_df, row["Player_1"], row["Surface"])
        df.at[idx, "player2_surface_winrate"] = calculate_surface_stats(temp_df, row["Player_2"], row["Surface"])
        # H2H features
        h2h_feats = calculate_h2h_features(df, row["Player_1"], row["Player_2"], up_to_date=row["Date"])
        df.at[idx, "h2h_matches"] = h2h_feats[0]
        df.at[idx, "h2h_wins_p1"] = h2h_feats[1]
        df.at[idx, "h2h_winrate_p1"] = h2h_feats[2]
        df.at[idx, "h2h_recent_p1_win"] = h2h_feats[3]
        df.at[idx, "h2h_streak_p1"] = h2h_feats[4]
    return df

# Add H2H columns before calling add_player_stats
df["h2h_matches"] = 0
df["h2h_wins_p1"] = 0
df["h2h_winrate_p1"] = 0.0
df["h2h_recent_p1_win"] = 0
df["h2h_streak_p1"] = 0

# Feature engineering
print("Calculating player statistics...")
df = create_features(df)
df = add_player_stats(df)
df = calculate_elo_ratings(df)

feature_columns = [
    "rank_diff", "odds_diff", "pts_diff", "odds_ratio", "prob_diff", "rank_ratio",
    "player1_form", "player2_form", "player1_surface_winrate", "player2_surface_winrate",
    "h2h_matches", "h2h_wins_p1", "h2h_winrate_p1", "h2h_recent_p1_win", "h2h_streak_p1",
    "elo_p1_pre", "elo_p2_pre", "elo_diff", "elo_win_prob_p1"
]

X = df[feature_columns].fillna(0)
y = df["target"]
X = X.replace([np.inf, -np.inf], 0)

if xgb_available:
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
else:
    model = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=8)
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Model ({'XGBoost' if xgb_available else 'GradientBoosting'}) CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

def get_latest_elo():
    elo_dict = {}
    for idx, row in df.sort_values('Date').iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        r1 = elo_dict.get(p1, 1500)
        r2 = elo_dict.get(p2, 1500)
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1
        s1 = 1 if row['Winner'] == p1 else 0
        s2 = 1 - s1
        r1_new = r1 + 32 * (s1 - e1)
        r2_new = r2 + 32 * (s2 - e2)
        elo_dict[p1] = r1_new
        elo_dict[p2] = r2_new
    return elo_dict

def predict_match_winner(player1_input, player2_input, surface="hard"):
    def get_best_match(name):
        matches = difflib.get_close_matches(name, player_names, n=1, cutoff=0.3)
        return matches[0] if matches else None
    player1 = get_best_match(player1_input)
    player2 = get_best_match(player2_input)
    if not player1 or not player2:
        return f"No close match found for: {'player1' if not player1 else ''} {'player2' if not player2 else ''}"
    def get_latest_stats(player):
        matches = df[(df["Player_1"] == player) | (df["Player_2"] == player)]
        if matches.empty:
            return None
        latest = matches.sort_values("Date", ascending=False).iloc[0]
        if latest["Player_1"] == player:
            return {
                "rank": latest["Rank_1"],
                "odds": latest["Odd_1"],
                "pts": latest["Pts_1"]
            }
        else:
            return {
                "rank": latest["Rank_2"],
                "odds": latest["Odd_2"],
                "pts": latest["Pts_2"]
            }
    def get_player_form(player, surface):
        player_matches = df[
            ((df["Player_1"] == player) | (df["Player_2"] == player)) &
            (df["Surface"] == surface)
        ]
        if player_matches.empty:
            return 0.5
        recent = player_matches.sort_values("Date", ascending=False).head(10)
        wins = len(recent[recent["Winner"] == player])
        return wins / len(recent) if len(recent) > 0 else 0.5
    stats1 = get_latest_stats(player1)
    stats2 = get_latest_stats(player2)
    if not stats1 or not stats2:
        return "Insufficient player data to predict"
    # Calculate features
    rank_diff = stats1["rank"] - stats2["rank"]
    odds_diff = stats1["odds"] - stats2["odds"]
    pts_diff = stats1["pts"] - stats2["pts"]
    odds_ratio = stats1["odds"] / stats2["odds"]
    implied_prob_1 = 1 / stats1["odds"]
    implied_prob_2 = 1 / stats2["odds"]
    prob_diff = implied_prob_1 - implied_prob_2
    rank_ratio = stats1["rank"] / stats2["rank"]
    player1_form = get_player_form(player1, surface)
    player2_form = get_player_form(player2, surface)
    player1_surface_winrate = calculate_surface_stats(df, player1, surface)
    player2_surface_winrate = calculate_surface_stats(df, player2, surface)
    h2h_feats = calculate_h2h_features(df, player1, player2)
    latest_elo = get_latest_elo()
    elo_p1 = latest_elo.get(player1, 1500)
    elo_p2 = latest_elo.get(player2, 1500)
    elo_diff = elo_p1 - elo_p2
    elo_win_prob_p1 = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
    features = np.array([[rank_diff, odds_diff, pts_diff, odds_ratio, prob_diff, rank_ratio,
        player1_form, player2_form, player1_surface_winrate, player2_surface_winrate] +
        h2h_feats + [elo_p1, elo_p2, elo_diff, elo_win_prob_p1]])
    proba = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    winner = player1 if pred == 1 else player2
    confidence = max(proba)
    insights = []
    if abs(rank_diff) > 50:
        insights.append(f"Significant ranking difference ({abs(rank_diff)} spots)")
    if abs(odds_diff) > 1:
        insights.append(f"Large odds difference ({abs(odds_diff):.2f})")
    if abs(player1_form - player2_form) > 0.3:
        insights.append(f"Recent form difference ({abs(player1_form - player2_form):.2f})")
    if abs(elo_diff) > 50:
        insights.append(f"Elo difference ({elo_diff:.0f})")
    result = f"Predicted winner: {winner} (Confidence: {confidence:.2%})\n"
    result += f"Model used: {'XGBoost' if xgb_available else 'GradientBoosting'}\n"
    result += f"Surface: {surface.capitalize()}\n"
    result += f"Elo: {player1}={elo_p1:.0f}, {player2}={elo_p2:.0f}\n"
    if insights:
        result += f"Key factors: {', '.join(insights)}"
    return result

if __name__ == "__main__":
    print("=== Universal Tennis Match Predictor ===\n")
    print("Available surfaces:", ", ".join([s.capitalize() for s in df["Surface"].unique()]))
    print("Available series:", ", ".join(df["Series"].unique()))
    print()
    p1 = input("Enter Player 1 name: ")
    p2 = input("Enter Player 2 name: ")
    surface = input("Enter surface (hard/grass/clay): ").lower()
    if surface not in ["hard", "grass", "clay"]:
        surface = "hard"
        print("Defaulting to hard surface")
    result = predict_match_winner(p1, p2, surface)
    print(f"\n{result}")
