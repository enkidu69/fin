import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, time

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "backtest_results_fin4.xlsx"
BACKTEST_MONTHS = 6

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

def detect_patterns(df_slice):
    """
    Identical logic to fin4.py
    """
    patterns = []
    if len(df_slice) < 5: return patterns
    
    # Define candles relative to slice end
    c0 = df_slice.iloc[-1] # Today
    c1 = df_slice.iloc[-2] # Yesterday
    c2 = df_slice.iloc[-3] # 2 days ago
    
    # Candle properties (Bodies)
    body0 = abs(c0['Close'] - c0['Open'])
    body1 = abs(c1['Close'] - c1['Open'])
    body2 = abs(c2['Close'] - c2['Open'])
    
    # Average body size for context (last 20 days)
    avg_body = (abs(df_slice['Close'] - df_slice['Open']).tail(20)).mean()
    
    # Shadows
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    lower1 = min(c1['Close'], c1['Open']) - c1['Low']
    
    # Trend Check (Crucial for Nison)
    if not get_trend(df_slice['Close']):
        return patterns

    # --- PATTERN LOGIC FROM FIN4.PY ---

    # 1. HAMMER
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0):
        patterns.append("Hammer")

    # 2. INVERTED HAMMER
    if (upper0 > 2 * body0) and (lower0 < 0.2 * body0):
        patterns.append("Inverted Hammer")

    # 3. BULLISH BELT-HOLD
    if (c0['Close'] > c0['Open']) and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5):
        patterns.append("Bullish Belt-Hold")

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

    # 9. MORNING STAR (STRICT)
    if (c2['Close'] < c2['Open'] and body2 > avg_body and          # 1. Long Red
        abs(c1['Close'] - c1['Open']) < body2 * 0.5 and            # 2. Small Star
        max(c1['Open'], c1['Close']) < c2['Close'] and             #    Star Top < Red Bottom (Gap)
        c0['Close'] > c0['Open'] and                               # 3. Green
        c0['Close'] > (c2['Close'] + body2/2)):                    #    Closes > 50% into 1st
        patterns.append("Morning Star")

    # 10. THREE WHITE SOLDIERS
    if (c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c0['Close'] > c0['Open'] and
        c0['Close'] > c1['Close'] > c2['Close'] and
        body0 > avg_body and body1 > avg_body and body2 > avg_body):
        patterns.append("Three White Soldiers")
        
    return patterns

def run_backtest():
    print(f"--- STRICT NISON BACKTESTER ({BACKTEST_MONTHS} Months) ---")
    
    # 1. Load Tickers
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
        print(f"Loaded {len(tickers)} tickers.")
    except Exception:
        print(f"Error: {INPUT_FILE} not found. Please create it.")
        return

    # 2. Download Data
    # Dataset A: Daily Data (For finding signals over 6 months)
    start_date_daily = datetime.now() - timedelta(days=(BACKTEST_MONTHS * 30) + 40)
    print("1/2 Downloading DAILY market data (6 months)...")
    daily_data = yf.download(tickers, start=start_date_daily, interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    
    # Dataset B: Intraday Data (For the "30 min" strategy)
    # LIMITATION: Yahoo only gives last 60 days of 30m data.
    print("2/2 Downloading INTRADAY (30m) market data (Last 60 days)...")
    try:
        intraday_data = yf.download(tickers, period="60d", interval="30m", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Intraday download warning: {e}")
        intraday_data = pd.DataFrame()

    trades = []
    
    # 3. Simulation Loop
    available_tickers = daily_data.columns.levels[0] if isinstance(daily_data.columns, pd.MultiIndex) else [tickers[0]] if len(tickers)==1 else []
    
    for ticker in available_tickers:
        # Extract Daily DF
        if isinstance(daily_data.columns, pd.MultiIndex):
            df = daily_data[ticker].dropna()
            df_intra = intraday_data[ticker].dropna() if (not intraday_data.empty and ticker in intraday_data.columns.levels[0]) else pd.DataFrame()
        else:
            df = daily_data.dropna()
            df_intra = intraday_data.dropna() if not intraday_data.empty else pd.DataFrame()

        if len(df) < 50: continue

        # Iterate days
        for i in range(30, len(df) - 1):
            
            # --- STRATEGY 1: SIGNAL DETECTION ---
            # Pretend it is 'simulation_date'
            simulation_date = df.index[i]
            df_slice = df.iloc[:i+1]
            
            found_patterns = detect_patterns(df_slice)
            
            if found_patterns:
                signal_price = df_slice.iloc[-1]['Close'] # Signal Close
                
                # Next Day Data
                next_day = df.iloc[i+1]
                next_date = df.index[i+1]
                next_close = next_day['Close']
                
                # Result 1: Daily Close-to-Close
                gain_pct_daily = ((next_close - signal_price) / signal_price) * 100
                
                # --- STRATEGY 2: INTRADAY (30m after Open -> 30m before Close) ---
                intra_buy = np.nan
                intra_sell = np.nan
                gain_pct_intra = np.nan
                intra_note = "Data Unavailable (>60d)"
                
                # Check if we have intraday data for this specific 'next_date'
                if not df_intra.empty:
                    # Filter for the specific date
                    day_bars = df_intra[df_intra.index.date == next_date.date()]
                    
                    if len(day_bars) >= 2:
                        # Logic:
                        # Buy at 10:00 AM (Open of the 2nd 30m bar, assuming 9:30 start)
                        # Sell at 3:30 PM (Open of the last 30m bar, assuming 4:00 close)
                        
                        try:
                            # Verify bars exist
                            # Bar 0: 09:30-10:00
                            # Bar 1: 10:00-10:30 (We buy at Open here)
                            intra_buy = day_bars.iloc[1]['Open']
                            
                            # Last Bar: 15:30-16:00 (We sell at Open here)
                            intra_sell = day_bars.iloc[-1]['Open']
                            
                            gain_pct_intra = ((intra_sell - intra_buy) / intra_buy) * 100
                            intra_note = "Executed"
                        except IndexError:
                            intra_note = "Partial Data"
                
                for pattern in found_patterns:
                    trades.append({
                        "Ticker": ticker,
                        "Signal Date": simulation_date.date(),
                        "Pattern": pattern,
                        "Next Date": next_date.date(),
                        
                        # Strategy 1 Results
                        "Signal Close": round(signal_price, 2),
                        "Next Close": round(next_close, 2),
                        "Daily Gain %": round(gain_pct_daily, 2),
                        "Daily Result": "WIN" if gain_pct_daily > 0 else "LOSS",
                        
                        # Strategy 2 Results
                        "Intraday Buy (10:00)": round(intra_buy, 2) if not pd.isna(intra_buy) else "N/A",
                        "Intraday Sell (15:30)": round(intra_sell, 2) if not pd.isna(intra_sell) else "N/A",
                        "Intraday Gain %": round(gain_pct_intra, 2) if not pd.isna(gain_pct_intra) else "N/A",
                        "Intraday Result": "WIN" if (not pd.isna(gain_pct_intra) and gain_pct_intra > 0) else ("LOSS" if not pd.isna(gain_pct_intra) else "N/A"),
                        "Intraday Note": intra_note
                    })

    # 4. Save
    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nAnalysis Complete. Results saved to {OUTPUT_FILE}")
        print(f"Total Signals: {len(results_df)}")
        
        # Stats
        wins_daily = len(results_df[results_df['Daily Result'] == 'WIN'])
        print(f"Strategy 1 (Daily) Win Rate: {(wins_daily/len(results_df))*100:.1f}%")
        
        # Intraday Stats (Only for rows where data was available)
        df_intra_valid = results_df[results_df['Intraday Note'] == 'Executed']
        if not df_intra_valid.empty:
            wins_intra = len(df_intra_valid[df_intra_valid['Intraday Result'] == 'WIN'])
            print(f"Strategy 2 (Intraday) Win Rate: {(wins_intra/len(df_intra_valid))*100:.1f}% (over {len(df_intra_valid)} trades)")
        else:
            print("Strategy 2: No recent trades found within the 60-day intraday limit.")
            
    else:
        print("No patterns found.")

if __name__ == "__main__":
    run_backtest()