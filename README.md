# Tennis_ML_Model

This project provides machine learning models to predict the outcomes of professional tennis matches using historical data and advanced statistical features. It includes two main scripts:

## 1. predictor.py
A universal tennis match predictor that uses a variety of features (ranking, odds, Elo ratings, player form, head-to-head, etc.) to predict the winner between any two players on a given surface.

### How it works
- Loads match data from `2024-2025.xlsx`.
- Engineers features such as ranking difference, odds, Elo ratings, player form, surface stats, and head-to-head records.
- Trains a machine learning model (XGBoost or Gradient Boosting) to predict match outcomes.
- Allows you to input two player names and a surface (hard/grass/clay) to get a prediction and confidence score.

### How to use
1. Make sure you have Python 3 and the required packages installed (see below).
2. Place `2024-2025.xlsx` in the same directory as the script.
3. Run the script:
   ```bash
   python predictor.py
   ```
4. Follow the prompts to enter player names and surface.

## 2. wimbledon_predictor.py
A specialized predictor for the Wimbledon tournament, focusing on grass court performance and best-of-5 match experience.

### How it works
- Loads and filters data for grass, outdoor, best-of-5 matches (Wimbledon style).
- Engineers features specific to grass court and Wimbledon performance.
- Trains multiple models (Gradient Boosting, Random Forest, XGBoost) and selects the best one.
- Simulates a tournament among top players and predicts the most likely Wimbledon champion.

### How to use
1. Ensure `2024-2025.xlsx` is present in the directory.
2. Run the script:
   ```bash
   python wimbledon_predictor.py
   ```
3. The script will print top contenders and the predicted Wimbledon winner.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost (optional, for best performance)

Install dependencies with:
```bash
pip install pandas numpy scikit-learn xgboost
```

## Data
- The model expects a file named `2024-2025.xlsx` with historical tennis match data, including columns for player names, surface, odds, rankings, and match results.

## Disclaimer
**This project is for educational and research purposes only. It is not intended for any financial, gambling, or betting activities. The predictions are not guaranteed and should not be used for making financial decisions. The authors and contributors of this project are not liable for any loss, damage, or consequences arising from the use of this software or its predictions. Use at your own risk.**
