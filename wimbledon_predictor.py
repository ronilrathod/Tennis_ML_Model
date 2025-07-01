import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WimbledonPredictor:
    def __init__(self, data_file="2024-2025.xlsx"):
        """Initialize the Wimbledon predictor with data"""
        self.df = pd.read_excel(data_file)
        self.prepare_data()
        self.train_models()
        
    def prepare_data(self):
        """Prepare and filter data for Wimbledon analysis"""
        print("Preparing Wimbledon-specific data...")
        
        # Basic preprocessing
        self.df.dropna(subset=["Winner", "Player_1", "Player_2"], inplace=True)
        self.df["Surface"] = self.df["Surface"].str.lower()
        self.df["Court"] = self.df["Court"].str.lower()
        self.df["Round"] = self.df["Round"].str.lower()
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d-%m-%y", errors='coerce')
        self.df = self.df.dropna(subset=["Date"])
        
        # Filter for Wimbledon-style matches (grass, outdoor, best of 5)
        self.wimbledon_data = self.df[
            (self.df["Surface"] == "grass") &
            (self.df["Court"] == "outdoor") &
            (self.df["Best of"] == 5)
        ].copy()
        
        # If not enough best-of-5 grass matches, include all grass matches
        if len(self.wimbledon_data) < 100:
            self.wimbledon_data = self.df[
                (self.df["Surface"] == "grass") &
                (self.df["Court"] == "outdoor")
            ].copy()
            print(f"Using all outdoor grass matches: {len(self.wimbledon_data)} matches")
        else:
            print(f"Using best-of-5 outdoor grass matches: {len(self.wimbledon_data)} matches")
        
        # Create features
        self.create_features()
        
    def create_features(self):
        """Create comprehensive features for Wimbledon prediction"""
        df = self.wimbledon_data.copy()
        
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
        
        # Add player statistics
        df = self.add_player_stats(df)
        
        self.wimbledon_data = df
        
    def add_player_stats(self, df):
        """Add player-specific statistics for grass court performance"""
        print("Calculating player grass court statistics...")
        
        # Initialize new columns
        df["player1_grass_form"] = 0.5
        df["player2_grass_form"] = 0.5
        df["player1_grass_winrate"] = 0.5
        df["player2_grass_winrate"] = 0.5
        df["player1_recent_form"] = 0.5
        df["player2_recent_form"] = 0.5
        df["player1_bo5_experience"] = 0
        df["player2_bo5_experience"] = 0
        
        # Calculate for each match
        for idx, row in df.iterrows():
            # Player form on grass (excluding current match)
            temp_df = df[df["Date"] < row["Date"]]
            
            # Grass-specific form (last 10 grass matches)
            df.at[idx, "player1_grass_form"] = self.calculate_grass_form(temp_df, row["Player_1"])
            df.at[idx, "player2_grass_form"] = self.calculate_grass_form(temp_df, row["Player_2"])
            
            # Overall grass win rate
            df.at[idx, "player1_grass_winrate"] = self.calculate_grass_winrate(temp_df, row["Player_1"])
            df.at[idx, "player2_grass_winrate"] = self.calculate_grass_winrate(temp_df, row["Player_2"])
            
            # Recent overall form (last 20 matches)
            df.at[idx, "player1_recent_form"] = self.calculate_recent_form(temp_df, row["Player_1"])
            df.at[idx, "player2_recent_form"] = self.calculate_recent_form(temp_df, row["Player_2"])
            
            # Best-of-5 experience
            df.at[idx, "player1_bo5_experience"] = self.calculate_bo5_experience(temp_df, row["Player_1"])
            df.at[idx, "player2_bo5_experience"] = self.calculate_bo5_experience(temp_df, row["Player_2"])
        
        # Add H2H features for each match
        df["h2h_matches"] = 0
        df["h2h_wins_p1"] = 0
        df["h2h_winrate_p1"] = 0.0
        df["h2h_recent_p1_win"] = 0
        df["h2h_streak_p1"] = 0
        for idx, row in df.iterrows():
            h2h_feats = self.calculate_h2h_features(df, row["Player_1"], row["Player_2"], up_to_date=row["Date"])
            df.at[idx, "h2h_matches"] = h2h_feats[0]
            df.at[idx, "h2h_wins_p1"] = h2h_feats[1]
            df.at[idx, "h2h_winrate_p1"] = h2h_feats[2]
            df.at[idx, "h2h_recent_p1_win"] = h2h_feats[3]
            df.at[idx, "h2h_streak_p1"] = h2h_feats[4]
        
        return df
    
    def calculate_grass_form(self, df, player_name, recent_matches=10):
        """Calculate player's recent form on grass"""
        grass_matches = df[
            ((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) &
            (df["Surface"] == "grass")
        ]
        
        if grass_matches.empty:
            return 0.5
        
        recent = grass_matches.sort_values("Date", ascending=False).head(recent_matches)
        wins = len(recent[recent["Winner"] == player_name])
        return wins / len(recent) if len(recent) > 0 else 0.5
    
    def calculate_grass_winrate(self, df, player_name):
        """Calculate player's overall win rate on grass"""
        grass_matches = df[
            ((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) &
            (df["Surface"] == "grass")
        ]
        
        if grass_matches.empty:
            return 0.5
        
        wins = len(grass_matches[grass_matches["Winner"] == player_name])
        return wins / len(grass_matches)
    
    def calculate_recent_form(self, df, player_name, recent_matches=20):
        """Calculate player's recent form across all surfaces"""
        player_matches = df[(df["Player_1"] == player_name) | (df["Player_2"] == player_name)]
        
        if player_matches.empty:
            return 0.5
        
        recent = player_matches.sort_values("Date", ascending=False).head(recent_matches)
        wins = len(recent[recent["Winner"] == player_name])
        return wins / len(recent) if len(recent) > 0 else 0.5
    
    def calculate_bo5_experience(self, df, player_name):
        """Calculate player's experience in best-of-5 matches"""
        bo5_matches = df[
            ((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) &
            (df["Best of"] == 5)
        ]
        
        return len(bo5_matches)
    
    def calculate_h2h_features(self, df, player1, player2, up_to_date=None):
        # Filter all matches between player1 and player2, optionally up to a certain date
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
        # Streak: count consecutive wins/losses for player1 from most recent
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
    
    def train_models(self):
        """Train prediction models"""
        print("Training Wimbledon prediction models...")
        
        # Features for the model
        self.feature_columns = [
            "rank_diff", "odds_diff", "pts_diff", "odds_ratio", "prob_diff", "rank_ratio",
            "player1_grass_form", "player2_grass_form", "player1_grass_winrate", "player2_grass_winrate",
            "player1_recent_form", "player2_recent_form", "player1_bo5_experience", "player2_bo5_experience",
            "h2h_matches", "h2h_wins_p1", "h2h_winrate_p1", "h2h_recent_p1_win", "h2h_streak_p1"
        ]
        
        # Prepare data
        X = self.wimbledon_data[self.feature_columns].fillna(0)
        y = self.wimbledon_data["target"]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Train models
        self.gb_model = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=6)
        self.rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        
        self.gb_model.fit(X, y)
        self.rf_model.fit(X, y)
        
        # Cross-validation scores
        gb_cv = cross_val_score(self.gb_model, X, y, cv=5)
        rf_cv = cross_val_score(self.rf_model, X, y, cv=5)
        
        print(f"Gradient Boosting CV Accuracy: {gb_cv.mean():.3f} (+/- {gb_cv.std() * 2:.3f})")
        print(f"Random Forest CV Accuracy: {rf_cv.mean():.3f} (+/- {rf_cv.std() * 2:.3f})")
        
        # Choose the better model
        if gb_cv.mean() > rf_cv.mean():
            self.best_model = self.gb_model
            self.model_name = "Gradient Boosting"
        else:
            self.best_model = self.rf_model
            self.model_name = "Random Forest"
        
        print(f"Selected model: {self.model_name}")
    
    def get_top_players(self, top_n=20):
        """Get top players based on grass court performance and recent form"""
        print(f"\nAnalyzing top {top_n} players for Wimbledon 2025...")
        
        # Get all unique players
        all_players = list(set(self.df["Player_1"].dropna().tolist() + self.df["Player_2"].dropna().tolist()))
        
        player_stats = []
        
        for player in all_players:
            # Get player's grass matches
            grass_matches = self.df[
                ((self.df["Player_1"] == player) | (self.df["Player_2"] == player)) &
                (self.df["Surface"] == "grass")
            ]
            
            if len(grass_matches) < 5:  # Need minimum matches
                continue
            
            # Calculate statistics
            grass_winrate = self.calculate_grass_winrate(self.df, player)
            recent_form = self.calculate_recent_form(self.df, player)
            grass_form = self.calculate_grass_form(self.df, player)
            bo5_experience = self.calculate_bo5_experience(self.df, player)
            
            # Get latest ranking
            player_matches = self.df[(self.df["Player_1"] == player) | (self.df["Player_2"] == player)]
            if not player_matches.empty:
                latest_match = player_matches.sort_values("Date", ascending=False).iloc[0]
                if latest_match["Player_1"] == player:
                    current_rank = latest_match["Rank_1"]
                else:
                    current_rank = latest_match["Rank_2"]
            else:
                current_rank = 999
            
            # Calculate Wimbledon score (weighted combination)
            wimbledon_score = (
                grass_winrate * 0.3 +
                grass_form * 0.25 +
                recent_form * 0.2 +
                (1 / (1 + current_rank/100)) * 0.15 +  # Ranking factor
                min(bo5_experience / 50, 1) * 0.1  # Experience factor
            )
            
            player_stats.append({
                "player": player,
                "grass_winrate": grass_winrate,
                "grass_form": grass_form,
                "recent_form": recent_form,
                "bo5_experience": bo5_experience,
                "current_rank": current_rank,
                "wimbledon_score": wimbledon_score,
                "grass_matches": len(grass_matches)
            })
        
        # Sort by Wimbledon score
        player_stats.sort(key=lambda x: x["wimbledon_score"], reverse=True)
        
        return player_stats[:top_n]
    
    def predict_wimbledon_winner(self, top_players=20):
        """Predict the 2025 Wimbledon winner"""
        print("=== 2025 Wimbledon Winner Prediction ===\n")
        
        # Get top players
        top_players_list = self.get_top_players(top_players)
        
        print("Top Contenders for Wimbledon 2025:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Player':<25} {'Grass WR':<8} {'Form':<6} {'Rank':<6} {'Score':<6}")
        print("-" * 80)
        
        for i, player in enumerate(top_players_list, 1):
            print(f"{i:<4} {player['player']:<25} {player['grass_winrate']:<8.3f} "
                  f"{player['grass_form']:<6.3f} {player['current_rank']:<6} {player['wimbledon_score']:<6.3f}")
        
        # Simulate tournament bracket
        print(f"\n🏆 Tournament Simulation Results:")
        print("=" * 50)
        
        # Use top 8 for final prediction
        finalists = top_players_list[:8]
        
        # Simulate matches between top players
        winner_predictions = []
        
        for i, player1 in enumerate(finalists):
            for j, player2 in enumerate(finalists[i+1:], i+1):
                # Create feature vector for this matchup
                features = self.create_matchup_features(player1, player2)
                
                if features is not None:
                    # Get prediction
                    proba = self.best_model.predict_proba([features])[0]
                    winner_prob = proba[1] if player1["wimbledon_score"] > player2["wimbledon_score"] else proba[0]
                    
                    winner = player1["player"] if winner_prob > 0.5 else player2["player"]
                    confidence = max(proba)
                    
                    winner_predictions.append({
                        "winner": winner,
                        "confidence": confidence,
                        "player1": player1["player"],
                        "player2": player2["player"]
                    })
        
        # Aggregate results
        winner_counts = {}
        for pred in winner_predictions:
            winner = pred["winner"]
            if winner not in winner_counts:
                winner_counts[winner] = {"wins": 0, "avg_confidence": 0, "total_confidence": 0}
            winner_counts[winner]["wins"] += 1
            winner_counts[winner]["total_confidence"] += pred["confidence"]
        
        # Calculate average confidence
        for winner in winner_counts:
            winner_counts[winner]["avg_confidence"] = winner_counts[winner]["total_confidence"] / winner_counts[winner]["wins"]
        
        # Sort by wins and confidence
        final_ranking = sorted(winner_counts.items(), 
                             key=lambda x: (x[1]["wins"], x[1]["avg_confidence"]), 
                             reverse=True)
        
        print("🏆 Final Prediction Rankings:")
        print("-" * 50)
        for i, (player, stats) in enumerate(final_ranking[:5], 1):
            print(f"{i}. {player} - Wins: {stats['wins']}, Avg Confidence: {stats['avg_confidence']:.3f}")
        
        # Final winner
        if final_ranking:
            champion = final_ranking[0][0]
            champion_stats = final_ranking[0][1]
            
            print(f"\n🎾 **2025 Wimbledon Champion Prediction:**")
            print(f"🏆 **{champion}**")
            print(f"   Tournament Wins: {champion_stats['wins']}")
            print(f"   Average Confidence: {champion_stats['avg_confidence']:.1%}")
            print(f"   Model Used: {self.model_name}")
            
            # Get champion details
            champion_data = next((p for p in top_players_list if p["player"] == champion), None)
            if champion_data:
                print(f"\n📊 Champion Statistics:")
                print(f"   Grass Court Win Rate: {champion_data['grass_winrate']:.1%}")
                print(f"   Recent Grass Form: {champion_data['grass_form']:.1%}")
                print(f"   Overall Recent Form: {champion_data['recent_form']:.1%}")
                print(f"   Current Ranking: #{champion_data['current_rank']}")
                print(f"   Best-of-5 Experience: {champion_data['bo5_experience']} matches")
        
        return champion if final_ranking else None
    
    def create_matchup_features(self, player1, player2):
        """Create feature vector for a matchup between two players"""
        try:
            rank_diff = player1["current_rank"] - player2["current_rank"]
            odds_diff = 0
            pts_diff = 0
            odds_ratio = 1.0
            rank_ratio = player1["current_rank"] / max(player2["current_rank"], 1)
            prob_diff = 0
            player1_grass_form = player1["grass_form"]
            player2_grass_form = player2["grass_form"]
            player1_grass_winrate = player1["grass_winrate"]
            player2_grass_winrate = player2["grass_winrate"]
            player1_recent_form = player1["recent_form"]
            player2_recent_form = player2["recent_form"]
            player1_bo5_experience = player1["bo5_experience"]
            player2_bo5_experience = player2["bo5_experience"]
            h2h_feats = self.calculate_h2h_features(self.df, player1["player"], player2["player"])
            features = [
                rank_diff, odds_diff, pts_diff, odds_ratio, prob_diff, rank_ratio,
                player1_grass_form, player2_grass_form, player1_grass_winrate, player2_grass_winrate,
                player1_recent_form, player2_recent_form, player1_bo5_experience, player2_bo5_experience
            ] + h2h_feats
            return features
        except Exception as e:
            print(f"Error creating features: {e}")
            return None

def main():
    """Main function to run Wimbledon prediction"""
    print("🎾 Wimbledon 2025 Winner Predictor")
    print("=" * 40)
    
    try:
        predictor = WimbledonPredictor()
        champion = predictor.predict_wimbledon_winner(top_players=20)
        
        if champion:
            print(f"\n🎯 **Final Prediction: {champion} will win Wimbledon 2025!**")
        else:
            print("\n❌ Unable to make prediction due to insufficient data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure the data file '2024-2025.xlsx' is in the same directory.")

if __name__ == "__main__":
    main() 