import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "ultimate_backtest_v13_PRO.xlsx"
BACKTEST_MONTHS = 12 
MIN_WIN_PCT = 0.02  # 0.02% threshold

# --- 1. INDICATORS (Updated with Professional Signals) ---
def calculate_indicators(df):
    df = df.copy()
    # SMA 20
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    std = df['Close'].rolling(window=20).std()
    df['LowerBB'] = df['SMA20'] - (2 * std)
    
    # Vol & Body
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()

    # --- NEW PROFESSIONAL INDICATORS ---
    
    # 1. MACD (12, 26, 9) - Trend Reversal Confirmation
    k_macd = df['Close'].ewm(span=12, adjust=False).mean()
    d_macd = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k_macd - d_macd
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 2. Stochastic Oscillator (14, 3, 3) - Oversold Reversal Entry
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    return df

def get_trend(row):
    if pd.isna(row['SMA20']): return False
    return row['Close'] < row['SMA20']

def is_doji(row, avg_body):
    return abs(row['Close'] - row['Open']) <= (avg_body * 0.1)

# --- 2. PATTERN DETECTION ---
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

# --- 3. STRATEGY ENGINE (User Requested) ---
def get_trade_advice(pattern, rsi, vol_ratio):
    p = pattern.lower()
    
    # Logic derived from your backtesting data
    if "hammer" in p and "inverted" not in p:
        if rsi < 30: return "STRONG BUY", "5 or 30 Days", "RSI < 30 confirms bottom"
        elif vol_ratio > 2.0: return "BUY", "30 Days", "High Volume support"
        else: return "WEAK", "-", "Needs RSI < 30"

    if "inverted hammer" in p:
        if vol_ratio > 2.0: return "BUY", "30 Days", "Volume Spike confirmed"
        else: return "WEAK", "-", "Fails without volume"

    if "engulfing" in p:
        if vol_ratio > 1.5: return "BUY", "30 Days", "Strong reversal candle"
        else: return "NEUTRAL", "30 Days", "Better with volume"

    if "tweezers" in p:
        if rsi < 35: return "BUY", "14 Days", "Good swing trade"
        else: return "WEAK", "-", "Needs oversold condition"

    if "harami" in p:
        if vol_ratio > 2.0: return "BUY", "5 Days", "Volume confirms brake"
        else: return "NEUTRAL", "5 Days", "Standard swing"

    if "belt-hold" in p: return "BUY", "30 Days", "High probability hold"
    if "morning star" in p: return "AVOID", "-", "Historically poor performance"

    return "OBSERVE", "-", "Standard pattern"

# --- MAIN ---
def run_backtest():
    print(f"--- ULTIMATE BACKTESTER V13 (Pro + Strat Engine) ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
        print(f"Loaded {len(tickers)} tickers.")
    except:
        print("Using demo tickers.")
        tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

    start_date = datetime.now() - timedelta(days=(BACKTEST_MONTHS * 30) + 120)
    
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
        
        for i in range(30, len(df) - 1):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            df_slice = df.iloc[i-5:i+1]
            found_patterns = detect_patterns(df_slice, current_row['AvgBody'])
            
            if found_patterns:
                signal_date = df.index[i]
                
                # ENTRY
                next_day_idx = i + 1
                next_day_row = df.iloc[next_day_idx]
                entry_price = next_day_row['Open']
                
                # --- STRATEGY A (1 Day / D+1) ---
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

                # --- MULTI-DAY HOLD STRATEGIES ---
                def get_trade_result(days_offset):
                    idx = next_day_idx + days_offset
                    if idx < len(df):
                        s_price = df.iloc[idx]['Close']
                        s_gain = ((s_price - entry_price) / entry_price) * 100
                        s_res = "WIN" if s_gain > MIN_WIN_PCT else "LOSS"
                        return s_price, s_gain, s_res
                    return None, None, "Open (Future)"

                # Short Term
                sell_F, gain_F, res_F = get_trade_result(2)
                sell_E, gain_E, res_E = get_trade_result(3)
                sell_G, gain_G, res_G = get_trade_result(4)
                sell_C, gain_C, res_C = get_trade_result(5)
                
                # Medium Term
                sell_H, gain_H, res_H = get_trade_result(6)
                sell_I, gain_I, res_I = get_trade_result(7)
                sell_J, gain_J, res_J = get_trade_result(8)
                sell_K, gain_K, res_K = get_trade_result(9)
                sell_L, gain_L, res_L = get_trade_result(10)
                sell_M, gain_M, res_M = get_trade_result(11)
                sell_N, gain_N, res_N = get_trade_result(12)
                sell_O, gain_O, res_O = get_trade_result(13)
                sell_P, gain_P, res_P = get_trade_result(14)
                
                # Long Term
                sell_D, gain_D, res_D = get_trade_result(30)

                # Expert Metrics
                rsi_val = current_row['RSI']
                is_oversold = rsi_val < 30
                vol_ratio = current_row['Volume'] / current_row['AvgVol'] if current_row['AvgVol'] > 0 else 0
                is_climax = vol_ratio > 2.0
                is_bb = (current_row['Low'] <= current_row['LowerBB'])

                # --- NEW PROFESSIONAL SIGNALS CHECK ---
                # MACD Cross: Current Hist > 0 and Previous Hist < 0
                macd_cross = "YES" if (current_row['MACD_Hist'] > 0 and prev_row['MACD_Hist'] < 0) else "NO"
                
                # Stochastic Cross: K crosses D while < 20 (Oversold)
                stoch_cross = "YES" if (current_row['Stoch_K'] > current_row['Stoch_D'] and 
                                        prev_row['Stoch_K'] < prev_row['Stoch_D'] and 
                                        current_row['Stoch_K'] < 20) else "NO"

                for pattern in found_patterns:
                    # CALL STRATEGY ENGINE
                    rec, time, rationale = get_trade_advice(pattern, rsi_val, vol_ratio)

                    trades.append({
                        "Ticker": ticker,
                        "Date": signal_date.date(),
                        "Pattern": pattern,
                        
                        # Strategy Engine Output
                        "Strategy Rec": rec,
                        "Rec Timeframe": time,
                        "Rationale": rationale,
                        
                        # Pro Signals
                        "MACD Buy Signal": macd_cross,
                        "Stoch Buy Signal": stoch_cross,

                        # Restored Columns
                        "Expert RSI": round(rsi_val, 1),
                        "RSI Status": "YES" if is_oversold else "NO",
                        "Rel Volume": round(vol_ratio, 1),
                        "Vol Status": "YES" if is_climax else "NO",
                        "BB Status": "YES" if is_bb else "NO",
                        
                        "Entry (Open)": round(entry_price, 2),
                        
                        "Strat A %": round(gain_A, 2), "Strat A Res": res_A,
                        "Strat B %": round(gain_B, 2) if gain_B else None, "Strat B Res": res_B,
                        
                        "Strat F %": round(gain_F, 2) if gain_F else None, "Strat F Res": res_F,
                        "Strat E %": round(gain_E, 2) if gain_E else None, "Strat E Res": res_E,
                        "Strat G %": round(gain_G, 2) if gain_G else None, "Strat G Res": res_G,
                        "Strat C %": round(gain_C, 2) if gain_C else None, "Strat C Res": res_C,
                        "Strat H %": round(gain_H, 2) if gain_H else None, "Strat H Res": res_H,
                        "Strat I %": round(gain_I, 2) if gain_I else None, "Strat I Res": res_I,
                        "Strat J %": round(gain_J, 2) if gain_J else None, "Strat J Res": res_J,
                        
                        "Strat K %": round(gain_K, 2) if gain_K else None, "Strat K Res": res_K,
                        "Strat L %": round(gain_L, 2) if gain_L else None, "Strat L Res": res_L,
                        "Strat M %": round(gain_M, 2) if gain_M else None, "Strat M Res": res_M,
                        "Strat N %": round(gain_N, 2) if gain_N else None, "Strat N Res": res_N,
                        "Strat O %": round(gain_O, 2) if gain_O else None, "Strat O Res": res_O,
                        "Strat P %": round(gain_P, 2) if gain_P else None, "Strat P Res": res_P,
                        
                        "Strat D %": round(gain_D, 2) if gain_D else None, "Strat D Res": res_D
                    })

    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nBacktest Complete. Results saved to {OUTPUT_FILE}")
        
        print("\n--- PERFORMANCE SUMMARY (Win Rates) ---")
        def print_stat(name, col):
            df_sub = results_df[results_df[col].isin(['WIN', 'LOSS'])]
            if not df_sub.empty:
                wins = len(df_sub[df_sub[col] == 'WIN'])
                print(f"{name}: {(wins/len(df_sub))*100:.1f}% ({wins}/{len(df_sub)})")
        
        print_stat("Scenario B (Intraday)", 'Strat B Res')
        print_stat("Scenario A (1 Day)", 'Strat A Res')
        print_stat("Scenario C (5 Days)", 'Strat C Res')
        print_stat("Scenario D (30 Days)", 'Strat D Res')
    else:
        print("No patterns found.")

if __name__ == "__main__":
    run_backtest()