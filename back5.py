import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


#results scenario on D+5


#Bullish Belt-Hold+ BB spring or RSI<30
#Bullish Engulfing+ vol climax
#Bullish Harami+ vol climax
#Bullish Harami Cross+ vol climax
#Hammer+ BB spring
#Tweezers Bottom 



# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "ultimate_backtest_results_v7.xlsx"
BACKTEST_MONTHS = 6 
MIN_WIN_PCT = 0.02  # 0.02% threshold for WIN

# --- INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    std = df['Close'].rolling(window=20).std()
    df['LowerBB'] = df['SMA20'] - (2 * std)
    
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    return df

def get_trend(row):
    if pd.isna(row['SMA20']): return False
    return row['Close'] < row['SMA20']

def is_doji(row, avg_body):
    return abs(row['Close'] - row['Open']) <= (avg_body * 0.1)

# --- PATTERNS ---
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

# --- MAIN ---
def run_backtest():
    print(f"--- ULTIMATE BACKTESTER V7 (5 Scenarios) ---")
    
    # 1. Load Tickers
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
        print(f"Loaded {len(tickers)} tickers.")
    except:
        print("Using demo tickers.")
        tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

    # 2. Download Data
    # Buffer for indicators + 30 days future
    start_date = datetime.now() - timedelta(days=(BACKTEST_MONTHS * 30) + 100)
    
    print("Step 1/2: Downloading DAILY data...")
    try:
        daily_data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("Step 2/2: Downloading INTRADAY (30m) data...")
    try:
        intraday_data = yf.download(tickers, period="60d", interval="30m", group_by='ticker', auto_adjust=True, threads=True)
    except:
        intraday_data = pd.DataFrame()

    trades = []
    
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
        
        df = calculate_indicators(df)
        
        # Loop
        for i in range(30, len(df) - 1):
            current_row = df.iloc[i]
            df_slice = df.iloc[i-5:i+1]
            found_patterns = detect_patterns(df_slice, current_row['AvgBody'])
            
            if found_patterns:
                signal_date = df.index[i]
                
                # ENTRY: Next Day Open (For A, C, D, E)
                next_day_idx = i + 1
                next_day_row = df.iloc[next_day_idx]
                entry_price = next_day_row['Open']
                
                # --- STRATEGY A (1 Day) ---
                sell_A = next_day_row['Close']
                gain_A = ((sell_A - entry_price) / entry_price) * 100
                res_A = "WIN" if gain_A > MIN_WIN_PCT else "LOSS"
                
                # --- STRATEGY B (Intraday) ---
                buy_B, sell_B, gain_B, res_B, note_B = None, None, None, None, "Data > 60d"
                next_date = df.index[next_day_idx]
                if not df_intra.empty:
                    if (datetime.now() - next_date).days < 58:
                        day_bars = df_intra[df_intra.index.date == next_date.date()]
                        if not day_bars.empty:
                            try:
                                # Fuzzy match 10:00 and 15:30
                                bw = day_bars.between_time("09:45", "10:15")
                                sw = day_bars.between_time("15:15", "15:45")
                                if not bw.empty and not sw.empty:
                                    buy_B = bw.iloc[0]['Open']
                                    sell_B = sw.iloc[-1]['Open']
                                    if pd.notna(buy_B) and pd.notna(sell_B):
                                        gain_B = ((sell_B - buy_B) / buy_B) * 100
                                        res_B = "WIN" if gain_B > MIN_WIN_PCT else "LOSS"
                                        note_B = "Executed"
                                    else: note_B = "NaN Data"
                                else: note_B = "Missing Window"
                            except: note_B = "Error"
                        else: note_B = "No Intraday Data"

                # --- STRATEGY E (3 Days) --- NEW
                idx_E = next_day_idx + 3
                if idx_E < len(df):
                    sell_E = df.iloc[idx_E]['Close']
                    gain_E = ((sell_E - entry_price) / entry_price) * 100
                    res_E = "WIN" if gain_E > MIN_WIN_PCT else "LOSS"
                else:
                    sell_E, gain_E, res_E = None, None, "Open Trade"

                # --- STRATEGY C (5 Days) ---
                idx_C = next_day_idx + 5
                if idx_C < len(df):
                    sell_C = df.iloc[idx_C]['Close']
                    gain_C = ((sell_C - entry_price) / entry_price) * 100
                    res_C = "WIN" if gain_C > MIN_WIN_PCT else "LOSS"
                else:
                    sell_C, gain_C, res_C = None, None, "Open Trade"

                # --- STRATEGY D (30 Days) ---
                idx_D = next_day_idx + 30
                if idx_D < len(df):
                    sell_D = df.iloc[idx_D]['Close']
                    gain_D = ((sell_D - entry_price) / entry_price) * 100
                    res_D = "WIN" if gain_D > MIN_WIN_PCT else "LOSS"
                else:
                    sell_D, gain_D, res_D = None, None, "Open Trade"

                # Expert Metrics
                rsi_val = current_row['RSI']
                is_oversold = rsi_val < 30
                vol_ratio = current_row['Volume'] / current_row['AvgVol'] if current_row['AvgVol'] > 0 else 0
                is_climax = vol_ratio > 2.0
                is_bb = (current_row['Low'] <= current_row['LowerBB'])

                for pattern in found_patterns:
                    trades.append({
                        "Ticker": ticker,
                        "Date": signal_date.date(),
                        "Pattern": pattern,
                        "Expert RSI": round(rsi_val, 1),
                        "RSI<30": "YES" if is_oversold else "NO",
                        "Vol Ratio": round(vol_ratio, 1),
                        "Vol Climax": "YES" if is_climax else "NO",
                        "BB Spring": "YES" if is_bb else "NO",
                        
                        "Entry Price (Open)": round(entry_price, 2),
                        
                        # Strat A (1d)
                        "Strat A Sell": round(sell_A, 2),
                        "Strat A Gain %": round(gain_A, 2),
                        "Strat A Result": res_A,
                        
                        # Strat B (Intra)
                        "Strat B Buy": round(buy_B, 2) if buy_B else None,
                        "Strat B Sell": round(sell_B, 2) if sell_B else None,
                        "Strat B Gain %": round(gain_B, 2) if gain_B else None,
                        "Strat B Result": res_B,
                        "Strat B Note": note_B,

                        # Strat E (3d)
                        "Strat E Sell (3d)": round(sell_E, 2) if sell_E else None,
                        "Strat E Gain %": round(gain_E, 2) if gain_E else None,
                        "Strat E Result": res_E,
                        
                        # Strat C (5d)
                        "Strat C Sell (5d)": round(sell_C, 2) if sell_C else None,
                        "Strat C Gain %": round(gain_C, 2) if gain_C else None,
                        "Strat C Result": res_C,
                        
                        # Strat D (30d)
                        "Strat D Sell (30d)": round(sell_D, 2) if sell_D else None,
                        "Strat D Gain %": round(gain_D, 2) if gain_D else None,
                        "Strat D Result": res_D
                    })

    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nBacktest Complete. Results saved to {OUTPUT_FILE}")
        
        print("\n--- PERFORMANCE SUMMARY ---")
        
        # Helper to calc win rate
        def print_win_rate(name, col):
            df_sub = results_df[results_df[col].isin(['WIN', 'LOSS'])]
            if not df_sub.empty:
                wins = len(df_sub[df_sub[col] == 'WIN'])
                print(f"{name}: {(wins/len(df_sub))*100:.1f}% ({wins}/{len(df_sub)})")
        
        print_win_rate("Scenario A (1 Day)", 'Strat A Result')
        print_win_rate("Scenario B (Intraday)", 'Strat B Result')
        print_win_rate("Scenario E (3 Days)", 'Strat E Result')
        print_win_rate("Scenario C (5 Days)", 'Strat C Result')
        print_win_rate("Scenario D (30 Days)", 'Strat D Result')

    else:
        print("No patterns found.")

if __name__ == "__main__":
    run_backtest()