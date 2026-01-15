import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "nison_expert_signals_pro_gonogo_v2.csv"

# --- 1. FUNDAMENTAL DATA FETCHER ---
def get_company_details(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "Sector": info.get('sector', 'N/A'),
            "Analyst Rec": info.get('recommendationKey', 'N/A').upper().replace('_', ' ') if info.get('recommendationKey') else 'N/A',
            "Target Price": info.get('targetMeanPrice', 'N/A')
        }
    except:
        return {"Sector": "N/A", "Analyst Rec": "N/A", "Target Price": "N/A"}

# --- 2. TECHNICAL CALCULATOR (Advanced) ---
def calculate_gonogo_trend(df):
    """
    Calculates the 'Go/No-Go' Trend and finds the START DATE of that trend.
    Logic:
    - GO: Price > EMA20 AND MACD Hist > 0 (Momentum aligned with Trend)
    - NO-GO: Price < EMA20 OR MACD Hist < 0
    """
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean() # Faster trend filter
    
    # 2. MACD (Standard 12, 26, 9)
    k_macd = df['Close'].ewm(span=12, adjust=False).mean()
    d_macd = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k_macd - d_macd
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 3. Strength (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- DEFINING THE STATUS ---
    # Strong Go: Price > EMA20, MACD Hist > 0, RSI > 50
    # Weak Go: Price > EMA20, MACD Hist > 0 (but RSI weak)
    # No Go: Anything else
    
    conditions = [
        (df['Close'] > df['EMA20']) & (df['MACD_Hist'] > 0) & (df['RSI'] > 50), # STRONG GO
        (df['Close'] > df['EMA20']) & (df['MACD_Hist'] > 0)                     # WEAK GO
    ]
    choices = [2, 1] # 2=Strong, 1=Weak, 0=NoGo
    
    df['Trend_Score'] = np.select(conditions, choices, default=0)
    
    return df

def find_trend_start(df):
    """
    Looks backwards from Today to find when the CURRENT status began.
    Returns: "Status (Date)"
    """
    if df.empty: return "N/A"
    
    current_score = df.iloc[-1]['Trend_Score']
    current_date = df.iloc[-1].name
    
    # Map score to text
    status_map = {2: "STRONG GO", 1: "WEAK GO", 0: "NO GO"}
    status_str = status_map.get(current_score, "NEUTRAL")
    
    # Iterate backwards to find the 'break' point
    start_date = current_date
    
    # Convert index to list for safe backwards iteration
    dates = df.index.tolist()
    scores = df['Trend_Score'].tolist()
    
    # Look back up to 150 days
    for i in range(len(scores) - 1, len(scores) - 150, -1):
        if i < 0: break
        if scores[i] != current_score:
            # The trend changed here, so the *next* day (i+1) was the start
            start_date = dates[i+1]
            break
        start_date = dates[i] # Keep pushing back if same

    return f"{status_str} ({start_date.strftime('%Y-%m-%d')})"

# --- 3. PATTERN & STRATEGY ---
def get_trade_advice(pattern, rsi):
    p = pattern.lower()
    if "hammer" in p and "inverted" not in p:
        return "BUY" if rsi < 30 else "NEUTRAL"
    if "engulfing" in p:
        return "BUY"
    return "OBSERVE"

def check_patterns_and_trend(ticker, df):
    if len(df) < 100: return None
    
    # Calculate Trend Data
    df = calculate_gonogo_trend(df)
    
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    
    # --- PATTERN DETECTION (Simplified for speed) ---
    patterns = []
    body0 = abs(c0['Close'] - c0['Open'])
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0): patterns.append("Hammer")
    if (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']): patterns.append("Bullish Engulfing")
    
    # Even if no pattern, we now capture the TREND STATUS
    trend_status_str = find_trend_start(df)
    
    # Pattern Logic
    pat_str = ", ".join(patterns) if patterns else "No Pattern"
    rating = get_trade_advice(pat_str, c0['RSI']) if patterns else "N/A"
    
    # Only return if there is a Pattern OR a 'GO' Trend
    # Remove this filter if you want ALL stocks regardless of status
    if not patterns and "NO GO" in trend_status_str:
        return None

    return {
        "Ticker": ticker,
        "Date": c0.name.strftime('%Y-%m-%d'),
        "Price": round(c0['Close'], 2),
        "Trend Status (Since)": trend_status_str, # The info you wanted
        "Pattern": pat_str,
        "Rating": rating,
        "RSI": round(c0['RSI'], 1),
        "MACD Hist": round(c0['MACD_Hist'], 3)
    }

# --- 4. HOURLY CHECK ---
def check_hourly_trend(ticker):
    """Checks the status on the 1H chart for the last business day."""
    try:
        df_h = yf.download(ticker, period="5d", interval="1h", progress=False, auto_adjust=True)
        if df_h.empty: return "No Data"
        
        # Reset Index & Clean
        df_h = df_h.reset_index()
        if isinstance(df_h.columns, pd.MultiIndex):
            df_h.columns = [c[0] if c[0] != 'Datetime' else 'Datetime' for c in df_h.columns]
            close_col = [c for c in df_h.columns if 'Close' in str(c)]
            if close_col: df_h['Close'] = df_h[close_col[0]]
            
        df_h = df_h.set_index('Datetime') # calc needs index
        df_h = calculate_gonogo_trend(df_h)
        
        # Get status of last bar
        last_bar = df_h.iloc[-1]
        score = last_bar['Trend_Score']
        status = "GO" if score > 0 else "NO GO"
        
        return f"{status} ({last_bar.name.strftime('%H:%M')})"
    except:
        return "Err"

def main():
    print("--- NISON EXPERT ADVISOR v5 (Trend Dating) ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print("Error: market_tickers.csv not found.")
        return

    print(f"Scanning {len(tickers)} stocks (1 Year Data)...")
    # Fetch 1y to ensure we can look back far enough for 'Since...' dates
    data = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, threads=True)
    
    results = []
    avail = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else []
    
    for ticker in avail:
        try:
            df_t = data[ticker].copy().dropna()
            res = check_patterns_and_trend(ticker, df_t)
            if res:
                print(f"{ticker}: {res['Trend Status (Since)']}")
                results.append(res)
        except: continue

    if results:
        print(f"\n--- FETCHING DETAILS ---")
        for res in results:
            # Add Fundamentals
            res.update(get_company_details(res['Ticker']))
            # Add Hourly
            res['Hourly Status'] = check_hourly_trend(res['Ticker'])
            
        df_res = pd.DataFrame(results)
        
        cols = [
            'Ticker', 'Date', 'Price', 
            'Trend Status (Since)', 'Hourly Status', # Key Info First
            'Pattern', 'Rating', 'RSI', 'MACD Hist',
            'Sector', 'Analyst Rec', 'Target Price'
        ]
        
        # Safe column filter
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]
        
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved to {OUTPUT_FILE}")
        print(df_res[['Ticker', 'Trend Status (Since)', 'Hourly Status']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()