import pandas as pd
import yfinance as yf
import numpy as np
import time
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "nison_expert_signals_dated.csv"

# --- 1. ROBUST DOWNLOADER ---
def safe_download(ticker, period, interval="1d"):
    """
    Downloads data with retry logic to handle Rate Limits.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Random delay to be polite
            time.sleep(random.uniform(0.1, 1.0)) 
            
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            
            if df.empty:
                return df
            
            # Clean MultiIndex if necessary
            if interval != "1d": 
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if c[0] != 'Datetime' else 'Datetime' for c in df.columns]
                    close_col = [c for c in df.columns if 'Close' in str(c)]
                    if close_col: df['Close'] = df[close_col[0]]
                df = df.set_index('Datetime')
                
            return df

        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait_time = random.uniform(20, 40)
                print(f"\n[!] Rate Limit Hit on {ticker}. Cooling down... (Attempt {attempt+1})")
                time.sleep(wait_time)
            else:
                return pd.DataFrame() 
    return pd.DataFrame()

def get_company_details(ticker):
    try:
        time.sleep(random.uniform(0.1, 0.5))
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "Sector": info.get('sector', 'N/A'),
            "Target Price": info.get('targetMeanPrice', 'N/A')
        }
    except:
        return {"Sector": "N/A", "Target Price": "N/A"}

# --- 2. TECHNICAL CALCULATOR ---
def calculate_technicals(df):
    df = df.copy()
    
    # Existing Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # NEW: Short-term momentum EMA
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    
    # MACD
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # NEW: Rate of Change (ROC) - 5 Day Momentum
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    
    # NEW: Average True Range (ATR) - Volatility measure
    df['Prev_Close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['Prev_Close']).abs()
    df['tr3'] = (df['Low'] - df['Prev_Close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df.drop(['Prev_Close', 'tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    
    # Bollinger Bands & NEW Z-Score
    std = df['Close'].rolling(window=20).std()
    sma20 = df['Close'].rolling(window=20).mean()
    df['UpperBB'] = df['EMA21'] + (2 * std)
    df['LowerBB'] = df['EMA21'] - (2 * std)
    df['BandWidth'] = (df['UpperBB'] - df['LowerBB']) / df['EMA21'].replace(0, np.nan)
    df['Z_Score'] = (df['Close'] - sma20) / std
    
    # Volume & Body Stats
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['AvgVol']
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    df['UpperShadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['LowerShadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    # NEW: Rolling VWAP (Volume Weighted Average Price) - 20 Day Proxy
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['Rolling_VWAP'] = (typical_price * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # NEW: Lagged ML Features
    df['Return_T1'] = df['Close'].pct_change(1)
    df['Return_T2'] = df['Close'].pct_change(2)

    # Trend Score (Go/No-Go)
    conditions = [
        (df['Close'] > df['EMA21']) & (df['MACD_Hist'] > 0) & (df['RSI'] > 50), 
        (df['Close'] > df['EMA21']) & (df['MACD_Hist'] > 0)
    ]
    choices = [2, 1]
    df['Trend_Score'] = np.select(conditions, choices, default=0)
    
    # Golden Cross
    df['Prev_SMA50'] = df['SMA50'].shift(1)
    df['Prev_SMA200'] = df['SMA200'].shift(1)
    df['Golden_Cross'] = (df['SMA50'] > df['SMA200']) & (df['Prev_SMA50'] <= df['Prev_SMA200'])
    
    return df

# --- 3. EXPERT SIGNALS ---

def check_buy_signal(df, squeeze_status):
    """
    NEW: Evaluates short-term momentum and volume to flag a 'Good Occasion to Buy'.
    """
    if len(df) < 20: return "NO"
    curr = df.iloc[-1]
    
    # 1. Momentum: Fast EMA is above Slow EMA, and 5-day rate of change is positive
    ema_cross_up = curr['EMA8'] > curr['EMA21']
    momentum_up = curr['ROC_5'] > 0
    
    # 2. RSI: Climbing but not overbought yet
    rsi_healthy = 45 < curr['RSI'] < 65
    
    # 3. Fuel: Price is above the 20-day VWAP, indicating buyers are in control
    above_vwap = curr['Close'] > curr['Rolling_VWAP']
    
    # 4. Catalyst: Higher than average volume OR a volatility squeeze
    catalyst = (curr['Vol_Ratio'] > 1.2) or ("YES" in squeeze_status)
    
    if ema_cross_up and momentum_up and rsi_healthy and above_vwap and catalyst:
        return "YES (Momentum + Fuel)"
    return "NO"

def check_divergence(df):
    if len(df) < 30: return "No"
    window = df.iloc[-30:]
    
    min_price_idx = window['Close'].idxmin()
    min_rsi_idx = window['RSI'].idxmin()
    curr = df.iloc[-1]
    
    if (curr.name - min_rsi_idx).days > 5 and (curr.name - min_price_idx).days < 3:
        if curr['RSI'] > window.loc[min_rsi_idx]['RSI']:
            return "YES (Bullish)"
    return "No"

def check_squeeze(df):
    if len(df) < 130: return "No"
    curr_width = df.iloc[-1]['BandWidth']
    six_month_min = df['BandWidth'].rolling(window=126).min().iloc[-1]
    
    if curr_width <= (six_month_min * 1.1):
        return "YES (Volatility Squeeze)"
    return "No"

def find_trend_status(df):
    if df.empty: return "N/A"
    current_score = df.iloc[-1]['Trend_Score']
    current_date = df.iloc[-1].name
    status_map = {2: "STRONG GO", 1: "WEAK GO", 0: "NO GO"}
    status_str = status_map.get(current_score, "NEUTRAL")
    
    start_date = current_date
    scores = df['Trend_Score'].tolist()
    dates = df.index.tolist()
    
    for i in range(len(scores) - 1, -1, -1):
        if scores[i] != current_score:
            start_date = dates[i+1]
            break
    return f"{status_str} (Since {start_date.strftime('%Y-%m-%d')})"

def check_ptj_rules(df):
    if df.empty or len(df) < 200: return "N/A", "N/A"
    curr = df.iloc[-1]
    ptj_status = "BULLISH (>200MA)" if curr['Close'] > curr['SMA200'] else "BEARISH (<200MA)"
    
    recent_cross = df.iloc[-20:]
    gc_date = "No"
    for idx, row in recent_cross.iterrows():
        if row['Golden_Cross']:
            gc_date = idx.strftime('%Y-%m-%d')
            break
    return ptj_status, gc_date

def check_patterns_full(ticker, df):
    if len(df) < 30: return "None", 0
    
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    c4 = df.iloc[-5]
    
    avg_body = c0['AvgBody']
    body0 = c0['Body']
    body1 = c1['Body']

    is_white = c0['Close'] > c0['Open']
    is_black = c0['Close'] < c0['Open']
    prev_white = c1['Close'] > c1['Open']
    prev_black = c1['Close'] < c1['Open']
    
    patterns = []
    
# 1. HAMMER
    if (c0['LowerShadow'] > 2 * body0) and (c0['UpperShadow'] < 0.2 * body0):
        if c1['Close'] < df.iloc[-10]['Close']:
            patterns.append("Hammer 10")

    if (c0['LowerShadow'] > 2 * body0) and (c0['UpperShadow'] < 0.2 * body0):
        if c0['Close'] < df.iloc[-3]['Close']:
            patterns.append("Hammer 3 remember low shadow")

    if (c0['UpperShadow'] > 2 * body0) and (c0['LowerShadow'] < 0.2 * body0):
        if c1['Close'] < df.iloc[-10]['Close']:
            patterns.append("Inverted Hammer not confirmed")
       
    # 2. INVERTED HAMMER
    if (c1['UpperShadow'] > 2 * body1) and (c1['LowerShadow'] < 0.2 * body1):
        if c0['Close'] < df.iloc[-3]['Close'] and c0['Close']>c1['Close']:
            patterns.append("Confirmed Inverted Hammer")

    # 3. DRAGONFLY DOJI
    is_doji = body0 <= (avg_body * 0.1)
    if is_doji and (c0['Open'] >= c0['High']*0.999) and (c0['LowerShadow'] > 2 * body0):
        patterns.append("Dragonfly Doji")

    # 4. BULLISH BELT-HOLD
    if is_white and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5):
        patterns.append("Bullish Belt-Hold")

    # 5. BULLISH ENGULFING
    if prev_black and is_white and (c0['Close'] > c1['Open']) and (c0['Open'] < c1['Close']) and c1['Close'] < df.iloc[-7]['Close']:
        patterns.append("Bullish Engulfing")

    # 6. BULLISH HARAMI
    if prev_black and is_white and (c0['Close'] < c1['Open']) and (c0['Open'] > c1['Close']):
        patterns.append("Bullish Harami")

    # 7. TWEEZERS BOTTOM
    if abs(c0['Low'] - c1['Low']) < (c0['Close'] * 0.002):
        patterns.append("Tweezers Bottom")

    # 8. BULLISH COUNTER ATTACK
    if prev_black and is_white and (c0['Open'] < c1['Low']):
        if abs(c0['Close'] - c1['Close']) < (c0['Close'] * 0.002):
            patterns.append("Counter Attack Bullish")
            
    # 9. BULLISH SEPARATING LINES
    if prev_black and is_white and abs(c0['Open'] - c1['Open']) < (c0['Close'] * 0.002):
        patterns.append("Bullish Separating Lines")

    # 10. RISING WINDOW
    if c0['Low'] > c1['High']:
        patterns.append("Rising Window")

    # 11. UPWARD GAPPING TASUKI
    if (c2['Close'] > c2['Open']) and prev_white and is_black:
        gap_exists = c1['Low'] > c2['High']
        opens_inside = (c0['Open'] < c1['Close']) and (c0['Open'] > c1['Open'])
        closes_in_gap = (c0['Close'] < c1['Open']) and (c0['Close'] > c2['High'])
        if gap_exists and opens_inside and closes_in_gap:
            patterns.append("Upward Gapping Tasuki")

    # 12. UPGAP SIDE-BY-SIDE WHITE LINES
    if prev_white and is_white:
        gap_exists = c1['Low'] > c2['High']
        similar_open = abs(c0['Open'] - c1['Open']) < (c0['Close'] * 0.002)
        if gap_exists and similar_open:
            patterns.append("Upgap Side-by-Side White Lines")

    # 13. HIGH PRICE GAPPING PLAY
    if c3['Low'] > c4['High']: 
        if abs(c1['Close'] - c2['Close']) < avg_body:
            patterns.append("High Price Gapping Play (Watch)")

    # 14. MORNING STAR
    if (c2['Close'] < c2['Open']) and (c2['Body'] > avg_body):
        if c1['Body'] < (avg_body * 0.6):
            if is_white and (c0['Close'] > (c2['Close'] + c2['Body']*0.5)):
                patterns.append("Morning Star pattern")

    # 15. RISING THREE METHODS
    if (c4['Close'] > c4['Open']) and (c4['Body'] > avg_body):
        if is_white and (c0['Close'] > c4['Close']):
            patterns.append("Rising Three Methods")

    # 16. FRYPAN BOTTOM
    small_bodies = all(df.iloc[-i]['Body'] < avg_body for i in range(2, 6))
    if small_bodies and is_white and (c0['Low'] > c1['High']):
        patterns.append("Frypan Bottom")

    # 17. TOWER BOTTOM
    if (c4['Close'] < c4['Open']) and (c4['Body'] > avg_body):
        consolidation = all(df.iloc[-i]['Body'] < avg_body for i in range(2, 5))
        if consolidation and is_white and (c0['Body'] > avg_body):
            patterns.append("Tower Bottom")

    # --- CONFIRMATIONS ---
    supports = []
    if (c0['Low'] < c0['SMA50']) and (c0['Close'] > c0['SMA50']): supports.append("50MA")
    if (c0['Low'] < c0['SMA200']) and (c0['Close'] > c0['SMA200']): supports.append("200MA")
    
    vol_conf = " (High Vol)" if c0['Volume'] > (c0['AvgVol'] * 1.5) else ""
    
    pattern_str = ", ".join(patterns)
    if supports and patterns:
        pattern_str += f" [Supp: {','.join(supports)}]"
    pattern_str += vol_conf

    return pattern_str if pattern_str else "None", c0['Vol_Ratio']

# --- 4. MAIN ENGINE ---
def main():
    print("--- NISON EXPERT SIGNALS (Updated with ML Features & Buy Logic) ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print("Error: market_tickers.csv not found.")
        return

    print(f"Scanning {len(tickers)} stocks...")
    
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return
    
    results = []
# NEW: Robust way to handle the yfinance column structure
    if isinstance(data.columns, pd.MultiIndex):
        # Find which level contains the tickers. Usually, it's the one that matches our input list.
        if tickers[0] in data.columns.get_level_values(0):
            avail = data.columns.levels[0]
            ticker_level = 0
        else:
            avail = data.columns.levels[1]
            ticker_level = 1
    else:
        avail = tickers if len(tickers) == 1 else []

    for ticker in avail:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # Extract data based on which level the ticker is on
                if ticker_level == 0:
                    df_t = data[ticker].copy().dropna()
                else:
                    df_t = data.xs(ticker, axis=1, level=1).copy().dropna()
            else:
                df_t = data.copy().dropna()

            if len(df_t) < 200: continue
            
            # 1. Calculate Technicals
            df_t = calculate_technicals(df_t)
            
            # 2. Run All Checks
            trend_status = find_trend_status(df_t)
            ptj_status, golden_cross = check_ptj_rules(df_t)
            patterns, vol_ratio = check_patterns_full(ticker, df_t)
            divergence = check_divergence(df_t)
            squeeze = check_squeeze(df_t)
            
            # NEW: Run Buy Signal Logic
            buy_signal = check_buy_signal(df_t, squeeze)
            
            # 3. Save Result
            c0 = df_t.iloc[-1]
            signal_date = c0.name.strftime('%Y-%m-%d')
            
            res = {
                "Date": signal_date,
                "Ticker": ticker,
                "Price": round(c0['Close'], 2),
                "Buy Signal": buy_signal,
                "Patterns": patterns,
                "Daily Trend": trend_status,
                "Bullish Divergence": divergence,
                "Volatility Squeeze": squeeze,
                "Golden Cross": golden_cross,
                "PTJ Status": ptj_status,
                "RSI": round(c0['RSI'], 1),
                "ROC(5)": round(c0['ROC_5'], 2),
                "Z-Score": round(c0['Z_Score'], 2),
                "Rel Vol": round(vol_ratio, 1)
            }
            results.append(res)
            
            if buy_signal != "NO" or patterns != "None":
                print(f"[{signal_date}] Alert: {ticker} -> Buy Signal: {buy_signal} | Pattern: {patterns}")
            
        except Exception as e:
            print(f"Error on {ticker}: {e}")
            continue

    if results:
        df_res = pd.DataFrame(results)
        # Reorder columns to put the most important stuff first
        cols = ['Date', 'Ticker', 'Price', 'Buy Signal', 'Patterns', 'ROC(5)', 'RSI', 'Z-Score', 'Rel Vol', 'Daily Trend', 'Bullish Divergence', 'Volatility Squeeze', 'PTJ Status', 'Golden Cross']
        df_res = df_res[[c for c in cols if c in df_res.columns]]
        print(df_res)
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\nCompleted. Saved all {len(results)} rows to {OUTPUT_FILE}")
        print(df_res.head(10).to_string(index=False))

if __name__ == "__main__":
    main()