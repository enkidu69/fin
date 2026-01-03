import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "nison_bullish_signals.csv"

def get_trend(series, window=20):
    """
    Steve Nison Rule: Reversals must occur in a downtrend.
    We define downtrend if current price is below the 20-period SMA.
    """
    if len(series) < window: return False
    sma = series.rolling(window=window).mean()
    curr = series.iloc[-1]
    return curr < sma.iloc[-1]

def is_doji(row, avg_body):
    """Returns True if the candle is a Doji (Open approx equal to Close)"""
    body = abs(row['Close'] - row['Open'])
    return body <= (avg_body * 0.1)

def check_patterns_for_ticker(ticker, df):
    """
    Checks a single dataframe (for one ticker) for patterns.
    """
    # Ensure enough data
    if len(df) < 20: return None
    
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['Open', 'Close', 'High', 'Low'])
    if len(df) < 4: return None

    # --- PRE-CALCULATIONS ---
    # We explicitly grab the DATE of the last candle
    last_date = df.index[-1]
    
    # Define candles (0 = Today/Latest, 1 = Yesterday, etc.)
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    
    # Candle properties (Bodies)
    body0 = abs(c0['Close'] - c0['Open'])
    body1 = abs(c1['Close'] - c1['Open'])
    body2 = abs(c2['Close'] - c2['Open'])
    
    # Average body size for context (last 20 days)
    avg_body = (abs(df['Close'] - df['Open']).tail(20)).mean()
    
    # Shadows
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    
    # Trend Check (Crucial for Nison)
    if not get_trend(df['Close']):
        return None

    patterns = []

    # ==========================================
    # SINGLE CANDLE PATTERNS
    # ==========================================
    
    # 1. HAMMER
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0):
        patterns.append("Hammer")

    # 2. INVERTED HAMMER
    if (upper0 > 2 * body0) and (lower0 < 0.2 * body0):
        patterns.append("Inverted Hammer")

    # 3. BULLISH BELT-HOLD
    if (c0['Close'] > c0['Open']) and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5):
        patterns.append("Bullish Belt-Hold")

    # ==========================================
    # TWO CANDLE PATTERNS
    # ==========================================

    # 4. BULLISH ENGULFING
    if (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and
        c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']):
        patterns.append("Bullish Engulfing")

    # 5. PIERCING PATTERN
    midpoint = c1['Close'] + (body1 / 2)
    if (c1['Close'] < c1['Open'] and body1 > avg_body and
        c0['Close'] > c0['Open'] and 
        c0['Open'] < c1['Low'] and       # Gap down open
        c0['Close'] > midpoint and       # Closes above midpoint
        c0['Close'] < c1['Open']):       # But stays inside body
        patterns.append("Piercing Pattern")

    # 6. BULLISH HARAMI (STRICT)
    if (c1['Close'] < c1['Open'] and body1 > avg_body and   # Prev Red
        c0['Close'] > c0['Open'] and                        # Curr Green
        c0['Close'] < c1['Open'] and                        # Green Top < Red Top
        c0['Open'] > c1['Close']):                          # Green Bottom > Red Bottom
        patterns.append("Bullish Harami")

    # 7. BULLISH HARAMI CROSS (STRICT)
    if (c1['Close'] < c1['Open'] and body1 > avg_body and
        is_doji(c0, avg_body) and 
        max(c0['Open'], c0['Close']) < c1['Open'] and 
        min(c0['Open'], c0['Close']) > c1['Close']):
        patterns.append("Bullish Harami Cross")

    # 8. TWEEZERS BOTTOM
    if abs(c0['Low'] - c1['Low']) < (c0['Low'] * 0.002):
        patterns.append("Tweezers Bottom")

    # ==========================================
    # THREE CANDLE PATTERNS
    # ==========================================

    # 10. MORNING STAR (STRICT)
    if (c2['Close'] < c2['Open'] and body2 > avg_body and          # 1. Long Red
        abs(c1['Close'] - c1['Open']) < body2 * 0.5 and            # 2. Small Star
        max(c1['Open'], c1['Close']) < c2['Close'] and             #    Star Top < Red Bottom (Gap)
        c0['Close'] > c0['Open'] and                               # 3. Green
        c0['Close'] > (c2['Close'] + body2/2)):                    #    Closes > 50% into 1st
        patterns.append("Morning Star")

    # 13. THREE WHITE SOLDIERS
    if (c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c0['Close'] > c0['Open'] and
        c0['Close'] > c1['Close'] > c2['Close'] and
        body0 > avg_body and body1 > avg_body and body2 > avg_body):
        patterns.append("Three White Soldiers")

    if patterns:
        return {
            "Ticker": ticker,
            "Date": last_date.strftime('%Y-%m-%d'),
            "Price": round(c0['Close'], 2),
            "Patterns": ", ".join(patterns),
            "Trend": "DOWN (Reversal Likely)",
            "Volume": c0.get('Volume', 0)
        }
    return None

def main():
    print("--- STEVE NISON BULLISH REVERSAL SCANNER ---")
    print("Reading market_tickers.csv...")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
    except FileNotFoundError:
        print("Error: market_tickers.csv not found.")
        return

    print(f"Batch downloading data for {len(tickers)} stocks...")
    
    try:
        # BATCH DOWNLOAD: Single request for all tickers to avoid 401 Error
        data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download failed: {e}")
        return
    
    print("\nScanning for patterns...")
    
    results = []
    
    # Handle Single Ticker vs Multiple Tickers Structure
    if len(tickers) == 1:
        # If single ticker, data columns are flat (Open, Close, etc.)
        res = check_patterns_for_ticker(tickers[0], data)
        if res: results.append(res)
    else:
        # If multiple, columns are MultiIndex (Ticker, Open)
        # We need to robustly handle cases where download might have failed for some
        available_tickers = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else []
        
        for ticker in available_tickers:
            try:
                df_ticker = data[ticker].copy()
                
                # Check if data is empty or all NaNs
                if df_ticker.empty or df_ticker['Close'].isna().all():
                    continue
                    
                res = check_patterns_for_ticker(ticker, df_ticker)
                if res:
                    print(f"FOUND: {res['Ticker']} [{res['Date']}] -> {res['Patterns']}")
                    results.append(res)
            except Exception as e:
                continue
    
    if results:
        df_res = pd.DataFrame(results)
        # Reorder columns to put Date first
        cols = ['Ticker', 'Date', 'Price', 'Patterns', 'Trend', 'Volume']
        df_res = df_res[cols]
        
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\nScan Complete. {len(df_res)} candidates found.")
        print(f"Results saved to {OUTPUT_FILE}")
        print(df_res)
    else:
        print("\nNo valid bullish reversal patterns found in downtrends.")

if __name__ == "__main__":
    main()