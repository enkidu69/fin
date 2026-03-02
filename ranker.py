import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. LOAD AND PREPARE DATA ---
print("Loading backtest data...")
try:
    df = pd.read_csv("core_pattern_performance.csv")
except FileNotFoundError:
    print("Error: core_pattern_performance.csv not found. Run the backtest first.")
    exit()

# Drop rows where D+5 hasn't happened yet (keeps the data clean)
df = df.dropna(subset=['D+5 Gain %'])

# --- 2. DEFINE THE TARGET (RISK-FIRST) ---
# Goal: Predict if the trade SURVIVED the -3% stop loss AND made a profit by D+5.
# 1 = Good Trade (Capital Preserved & Profitable), 0 = Bad Trade (Hit Stop or Lost Money)
df['Target_Success'] = ((df['Hit -3% Stop?'] == 'NO') & (df['D+5 Gain %'] > 0)).astype(int)

# --- 3. FEATURE ENGINEERING ---
# Convert the text-based 'Signals Triggered' into binary columns (One-Hot Encoding)
# e.g., if a row has "Hammer 10 | Momentum Buy Setup", it gets a 1 in both columns.
# --- 3. FEATURE ENGINEERING (FIXED) ---
# Safely clean and split the patterns without shredding the words
df['Clean_Signals'] = df['Signals Triggered'].astype(str).str.replace(" | ", "|", regex=False)
patterns = df['Clean_Signals'].str.get_dummies(sep='|')



# Combine the pattern columns with the numerical indicators
try:
    features = pd.concat([df[['RSI', 'ROC_5', 'Vol_Ratio']], patterns], axis=1)
except KeyError as e:
    print(f"Missing column in CSV: {e}. Did you add RSI, ROC_5, and Vol_Ratio to the backtest export?")
    exit()

X = features.fillna(0) # Handle any missing data
y = df['Target_Success']

# --- 4. TRAIN/TEST SPLIT ---
# We hold back 20% of the data. The model never sees this during training to prevent "memorizing" the answers.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. TRAIN THE RANDOM FOREST ---
print("Training Random Forest Classifier on historical setups...\n")
# We limit the depth (max_depth=5) to prevent 'overfitting' to market noise.
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# --- 6. TEST AND EVALUATE ---
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Unseen Data: {accuracy * 100:.2f}%\n")
print("*Note: In quantitative finance, an accuracy of 55-60% combined with strict stop-losses is a highly profitable system.*\n")

# --- 7. EXTRACT FEATURE IMPORTANCE ---
# This is where the machine tells us what actually matters
importances = rf_model.feature_importances_
feature_names = X.columns
feature_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances * 100
}).sort_values(by='Importance', ascending=False)

print("--- WHAT ACTUALLY DRIVES A WINNING TRADE? ---")
print("(Ranked by Mathematical Importance)")
print(feature_ranking.to_string(index=False, float_format="%.2f%%"))