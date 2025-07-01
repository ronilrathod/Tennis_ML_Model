import pandas as pd
import difflib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_excel("2024-2025.xlsx")

# Get all unique player names
player_names = list(set(df["Player_1"].dropna().tolist() + df["Player_2"].dropna().tolist()))

# Preprocessing
df.dropna(subset=["Winner", "Player_1", "Player_2"], inplace=True)
df["Surface"] = df["Surface"].str.lower()
df["Court"] = df["Court"].str.lower()
df["Round"] = df["Round"].str.lower()
df["Series"] = df["Series"].str.lower()

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y", errors='coerce')
df = df.dropna(subset=["Date"])

# Feature engineering
def create_features(df):
    """Create comprehensive features for prediction"""
    
    # Basic differences
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["odds_diff"] = df["Odd_1"] - df["Odd_2"]
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
    
    # Odds ratios and probabilities
    df["odds_ratio"] = df["Odd_1"] / df["Odd_2"]
    df["implied_prob_1"] = 1 / df["Odd_1"]
    df["implied_prob_2"] = 1 / df["Odd_2"]
    df["prob_diff"] = df["implied_prob_1"] - df["implied_prob_2"]
    
    # Rank ratios
    df["rank_ratio"] = df["Rank_1"] / df["Rank_2"]
    
    # Target variable
    df["target"] = (df["Winner"] == df["Player_1"]).astype(int)
    
    return df

# Create features
df = create_features(df)

# Player form analysis
def calculate_player_form(df, player_name, surface=None, recent_matches=10):
    """Calculate player's recent form"""
    player_matches = df[(df["Player_1"] == player_name) | (df["Player_2"] == player_name)]
    
    if surface:
        player_matches = player_matches[player_matches["Surface"] == surface]
    
    if player_matches.empty:
        return 0.5  # Neutral form if no matches
    
    # Get recent matches
    recent = player_matches.sort_values("Date", ascending=False).head(recent_matches)
    
    wins = 0
    total_matches = len(recent)
    
    for _, match in recent.iterrows():
        if match["Winner"] == player_name:
            wins += 1
    
    return wins / total_matches if total_matches > 0 else 0.5

# Surface-specific win rates
def calculate_surface_stats(df, player_name, surface):
    """Calculate player's win rate on specific surface"""
    surface_matches = df[
        ((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) &
        (df["Surface"] == surface)
    ]
    
    if surface_matches.empty:
        return 0.5
    
    wins = len(surface_matches[surface_matches["Winner"] == player_name])
    total = len(surface_matches)
    
    return wins / total

# Enhanced feature engineering with player statistics
def add_player_stats(df):
    """Add player-specific statistics as features"""
    
    # Initialize new columns
    df["player1_form"] = 0.5
    df["player2_form"] = 0.5
    df["player1_surface_winrate"] = 0.5
    df["player2_surface_winrate"] = 0.5
    df["player1_recent_rank"] = df["Rank_1"]
    df["player2_recent_rank"] = df["Rank_2"]
    
    # Calculate for each match
    for idx, row in df.iterrows():
        # Player form (excluding current match)
        temp_df = df[df["Date"] < row["Date"]]
        
        df.at[idx, "player1_form"] = calculate_player_form(temp_df, row["Player_1"], row["Surface"])
        df.at[idx, "player2_form"] = calculate_player_form(temp_df, row["Player_2"], row["Surface"])
        
        # Surface-specific win rates
        df.at[idx, "player1_surface_winrate"] = calculate_surface_stats(temp_df, row["Player_1"], row["Surface"])
        df.at[idx, "player2_surface_winrate"] = calculate_surface_stats(temp_df, row["Player_2"], row["Surface"])
    
    return df

# Add player statistics
print("Calculating player statistics...")
df = add_player_stats(df)

# Create surface-specific models
surfaces = df["Surface"].unique()
surface_models = {}

# Features for the model
feature_columns = [
    "rank_diff", "odds_diff", "pts_diff", "odds_ratio", "prob_diff", "rank_ratio",
    "player1_form", "player2_form", "player1_surface_winrate", "player2_surface_winrate"
]

# Train models for each surface
for surface in surfaces:
    surface_data = df[df["Surface"] == surface].copy()
    
    if len(surface_data) < 50:  # Skip surfaces with too few matches
        continue
    
    # Prepare features
    X = surface_data[feature_columns].fillna(0)
    y = surface_data["target"]
    
    # Remove rows with infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    if len(X) < 20:  # Need minimum data
        continue
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=6)
    
    try:
        model.fit(X, y)
        surface_models[surface] = model
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"{surface.capitalize()} surface model - CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    except:
        continue

# Global model (all surfaces combined)
print("\nTraining global model...")
X_global = df[feature_columns].fillna(0)
X_global = X_global.replace([np.inf, -np.inf], 0)
y_global = df["target"]

global_model = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=8)
global_model.fit(X_global, y_global)

cv_scores_global = cross_val_score(global_model, X_global, y_global, cv=5)
print(f"Global model - CV Accuracy: {cv_scores_global.mean():.3f} (+/- {cv_scores_global.std() * 2:.3f})")

def predict_match_winner(player1_input, player2_input, surface="hard", tournament_series="atp250"):
    """
    Enhanced prediction function with surface-specific models and comprehensive features
    """
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
        """Get player's recent form on specific surface"""
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
    
    # Player form
    player1_form = get_player_form(player1, surface)
    player2_form = get_player_form(player2, surface)
    
    # Surface win rates
    player1_surface_winrate = calculate_surface_stats(df, player1, surface)
    player2_surface_winrate = calculate_surface_stats(df, player2, surface)

    # Prepare feature vector
    features = np.array([[
        rank_diff, odds_diff, pts_diff, odds_ratio, prob_diff, rank_ratio,
        player1_form, player2_form, player1_surface_winrate, player2_surface_winrate
    ]])

    # Make prediction using surface-specific model if available, otherwise global model
    if surface in surface_models:
        model = surface_models[surface]
        proba = model.predict_proba(features)[0]
        pred = model.predict(features)[0]
        model_type = f"{surface.capitalize()}-specific"
    else:
        model = global_model
        proba = model.predict_proba(features)[0]
        pred = model.predict(features)[0]
        model_type = "Global"

    winner = player1 if pred == 1 else player2
    confidence = max(proba)

    # Additional insights
    insights = []
    if abs(rank_diff) > 50:
        insights.append(f"Significant ranking difference ({abs(rank_diff)} spots)")
    if abs(odds_diff) > 1:
        insights.append(f"Large odds difference ({abs(odds_diff):.2f})")
    if abs(player1_form - player2_form) > 0.3:
        insights.append(f"Recent form difference ({abs(player1_form - player2_form):.2f})")

    result = f"Predicted winner: {winner} (Confidence: {confidence:.2%})\n"
    result += f"Model used: {model_type}\n"
    result += f"Surface: {surface.capitalize()}\n"
    
    if insights:
        result += f"Key factors: {', '.join(insights)}"

    return result

# Example use
if __name__ == "__main__":
    print("=== Enhanced Tennis Match Predictor ===\n")
    print("Available surfaces:", ", ".join([s.capitalize() for s in surfaces]))
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
