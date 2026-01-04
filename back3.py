import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "ultimate_backtest_results_v4.xlsx"
BACKTEST_MONTHS = 36 

# --- TECHNICAL INDICATORS ---
def calculate_indicators(df):
    """Pre-calculates indicators for the entire history."""
    df = df.copy()
    
    # 1. SMA 20
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (20, 2)
    std = df['Close'].rolling(window=20).std()
    df['LowerBB'] = df['SMA20'] - (2 * std)
    
    # 4. Average Volume (20)
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    
    # 5. Average Body (20)
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    
    return df

def get_trend(row):
    if pd.isna(row['SMA20']): return False
    return row['Close'] < row['SMA20']

def is_doji(row, avg_body):
    return abs(row['Close'] - row['Open']) <= (avg_body * 0.1)

# --- PATTERN LOGIC ---
def detect_patterns(df_slice, avg_body):
    patterns = []
    if len(df_slice) < 5: return patterns
    
    c0 = df_slice.iloc[-1]
    c1 = df_slice.iloc[-2]
    c2 = df_slice.iloc[-3]
    
    body0 = abs(c0['Close'] - c0['Open'])
    body1 = abs(c1['Close'] - c1['Open'])
    body2 = abs(c2['Close'] - c2['Open'])
    
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    
    if not get_trend(c0): return patterns

    # Patterns
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0): patterns.append("Hammer")
    if (upper0 > 2 * body0) and (lower0 < 0.2 * body0): patterns.append("Inverted Hammer")
    if (c0['Close'] > c0['Open']) and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5): patterns.append("Bullish Belt-Hold")
    if (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']): patterns.append("Bullish Engulfing")
    
    midpoint = c1['Close'] + (body1 / 2)
    if (c1['Close'] < c1['Open'] and body1 > avg_body and c0['Close'] > c0['Open'] and c0['Open'] < c1['Low'] and c0['Close'] > midpoint and c0['Close'] < c1['Open']): patterns.append("Piercing Pattern")
    
    if (c1['Close'] < c1['Open'] and body1 > avg_body and c0['Close'] > c0['Open'] and c0['Close'] < c1['Open'] and c0['Open'] > c1['Close']): patterns.append("Bullish Harami")
    
    if (c1['Close'] < c1['Open'] and body1 > avg_body and is_doji(c0, avg_body) and max(c0['Open'], c0['Close']) < c1['Open'] and min(c0['Open'], c0['Close']) > c1['Close']): patterns.append("Bullish Harami Cross")
    
    if abs(c0['Low'] - c1['Low']) < (c0['Low'] * 0.002): patterns.append("Tweezers Bottom")
    
    if (c2['Close'] < c2['Open'] and body2 > avg_body and abs(c1['Close'] - c1['Open']) < body2 * 0.5 and max(c1['Open'], c1['Close']) < c2['Close'] and c0['Close'] > c0['Open'] and c0['Close'] > (c2['Close'] + body2/2)): patterns.append("Morning Star")
    
    if (c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Close'] > c2['Close'] and body0 > avg_body and body1 > avg_body and body2 > avg_body): patterns.append("Three White Soldiers")
        
    return patterns

# --- MAIN BACKTESTER ---
def run_backtest():
    print(f"--- ULTIMATE BACKTESTER V4 (Flexible Window) ---")
    
    # 1. Load Tickers
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
        print(f"Loaded {len(tickers)} tickers.")
    except:
        print(f"Error: {INPUT_FILE} not found. Using defaults.")
        tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

    # 2. Download Data
    start_date = datetime.now() - timedelta(days=(BACKTEST_MONTHS * 30) + 60)
    
    print("Step 1/2: Downloading DAILY data...")
    try:
        daily_data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return
        
    print("Step 2/2: Downloading INTRADAY (30m) data...")
    try:
        # Fetch intraday data. NOTE: Yahoo only gives last 60 days.
        intraday_data = yf.download(tickers, period="60d", interval="30m", group_by='ticker', auto_adjust=True, threads=True)
    except Exception:
        intraday_data = pd.DataFrame()

    trades = []
    
    # 3. Process Tickers
    available_tickers = daily_data.columns.levels[0] if isinstance(daily_data.columns, pd.MultiIndex) else [tickers[0]] if len(tickers)==1 else []
    
    for ticker in available_tickers:
        if isinstance(daily_data.columns, pd.MultiIndex):
            df = daily_data[ticker].copy()
            df_intra = intraday_data[ticker].copy() if (not intraday_data.empty and ticker in intraday_data.columns.levels[0]) else pd.DataFrame()
        else:
            df = daily_data.copy()
            df_intra = intraday_data.copy() if not intraday_data.empty else pd.DataFrame()

        df = df.dropna(how='all')
        if len(df) < 50: continue
        
        # Indicators
        df = calculate_indicators(df)
        
        # 4. Simulation
        for i in range(30, len(df) - 1):
            current_row = df.iloc[i]
            df_slice = df.iloc[i-5:i+1]
            
            # Detect
            found_patterns = detect_patterns(df_slice, current_row['AvgBody'])
            
            if found_patterns:
                signal_date = df.index[i]
                
                # --- Strategy A ---
                next_day = df.iloc[i+1]
                next_date = df.index[i+1]
                
                buy_A = next_day['Open']
                sell_A = next_day['Close']
                gain_A = ((sell_A - buy_A) / buy_A) * 100
                res_A = "WIN" if gain_A > 0 else "LOSS"
                
                # --- Strategy B (Flexible Window) ---
                buy_B = None
                sell_B = None
                gain_B = None
                res_B = None
                note_B = "Data > 60d" 
                
                if not df_intra.empty:
                    # Check if date is within last 60 days
                    days_diff = (datetime.now() - next_date).days
                    if days_diff < 58:
                        # Extract just the day's bars
                        day_bars = df_intra[df_intra.index.date == next_date.date()]
                        
                        if not day_bars.empty:
                            try:
                                # FLEXIBLE BUY: Look for any bar between 09:45 and 10:15
                                # We take the FIRST bar found in this window.
                                buy_window = day_bars.between_time("09:45", "10:15")
                                
                                # FLEXIBLE SELL: Look for any bar between 15:15 and 15:45
                                # We take the LAST bar found in this window.
                                sell_window = day_bars.between_time("15:15", "15:45")
                                
                                if not buy_window.empty and not sell_window.empty:
                                    b_price = buy_window.iloc[0]['Open']
                                    s_price = sell_window.iloc[-1]['Open']
                                    
                                    if pd.notna(b_price) and pd.notna(s_price):
                                        buy_B = b_price
                                        sell_B = s_price
                                        gain_B = ((sell_B - buy_B) / buy_B) * 100
                                        res_B = "WIN" if gain_B > 0 else "LOSS"
                                        note_B = "Executed"
                                    else:
                                        note_B = "Incomplete Data (NaN)"
                                else:
                                    note_B = "Missing Time Window (e.g. half day?)"
                            except:
                                note_B = "Error"
                        else:
                            note_B = "No Intraday Data for Date"
                
                # Expert Metrics
                rsi_val = current_row['RSI']
                vol_ratio = current_row['Volume'] / current_row['AvgVol'] if current_row['AvgVol'] > 0 else 0
                is_bb_spring = (current_row['Low'] <= current_row['LowerBB'])
                
                for pattern in found_patterns:
                    trades.append({
                        "Ticker": ticker,
                        "Date": signal_date.date(),
                        "Pattern": pattern,
                        "Expert RSI": round(rsi_val, 1),
                        "RSI<30": "YES" if rsi_val < 30 else "NO",
                        "Vol Ratio": round(vol_ratio, 1),
                        "Vol Climax": "YES" if vol_ratio > 2.0 else "NO",
                        "BB Spring": "YES" if is_bb_spring else "NO",
                        
                        "Strat A Buy": round(buy_A, 2),
                        "Strat A Sell": round(sell_A, 2),
                        "Strat A Gain %": round(gain_A, 2),
                        "Strat A Result": res_A,
                        
                        "Strat B Buy": round(buy_B, 2) if buy_B else None,
                        "Strat B Sell": round(sell_B, 2) if sell_B else None,
                        "Strat B Gain %": round(gain_B, 2) if gain_B else None,
                        "Strat B Result": res_B,
                        "Strat B Note": note_B
                    })

    # 5. Results
    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nBacktest Complete. Results saved to {OUTPUT_FILE}")
        
        print("\n--- PERFORMANCE SUMMARY ---")
        wins_A = len(results_df[results_df['Strat A Result'] == 'WIN'])
        print(f"Strategy A (Swing) Win Rate: {(wins_A/len(results_df))*100:.1f}% ({wins_A}/{len(results_df)})")
        
        # Strategy B Stats
        df_B = results_df[results_df['Strat B Result'].notna()].copy()
        
        if not df_B.empty:
            wins_B = len(df_B[df_B['Strat B Result'] == 'WIN'])
            print(f"Strategy B (Intraday) Win Rate: {(wins_B/len(df_B))*100:.1f}% ({wins_B}/{len(df_B)})")
            
            df_A_sub = results_df.loc[df_B.index]
            wins_A_sub = len(df_A_sub[df_A_sub['Strat A Result'] == 'WIN'])
            print(f"   (vs Strat A on same days: {(wins_A_sub/len(df_B))*100:.1f}%)")
        else:
            print("Strategy B: No recent trades available (limited to last 60 days).")
            
    else:
        print("No patterns found.")

if __name__ == "__main__":
    run_backtest()