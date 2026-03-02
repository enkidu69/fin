import pandas as pd
import yfinance as yf
import numpy as np
import time
import random

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE_BACKTEST = "vault_door_performance.csv"

BACKTEST_MONTHS = 12 
FEE_BPS = 15 # Standard broker execution overhead

def calculate_technicals_and_patterns(df):
    df = df.copy()
    
    # Standard Indicators
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    
    std = df['Close'].rolling(window=20).std()
    df['BandWidth'] = (4 * std) / df['EMA21'].replace(0, np.nan)
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['AvgVol']
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['Rolling_VWAP'] = (typical_price * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Calculate ATR (Average True Range) for the Vault Door
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # --- VECTORIZED CANDLESTICKS (NISON CORE) ---
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    df['UpperShadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['LowerShadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    is_white = df['Close'] > df['Open']
    is_black = df['Close'] < df['Open']
    lags = {i: df.shift(i) for i in range(1, 11)}
    prev_black = lags[1]['Close'] < lags[1]['Open']
    
    df['Patterns_Triggered'] = ""
    def add_pattern(condition, name):
        df.loc[condition, 'Patterns_Triggered'] += name + " | "

    # 1. HAMMER 10
    cond_h10 = (df['LowerShadow'] > 2 * df['Body']) & (df['UpperShadow'] < 0.2 * df['Body']) & (lags[1]['Close'] < lags[10]['Close'])
    add_pattern(cond_h10, "Hammer 10")

    # 2. INVERTED HAMMER
    cond_ih = (lags[1]['UpperShadow'] > 2 * lags[1]['Body']) & (lags[1]['LowerShadow'] < 0.2 * lags[1]['Body']) & (df['Close'] < lags[3]['Close']) & (df['Close'] > lags[1]['Close'])
    add_pattern(cond_ih, "Confirmed Inverted Hammer")

    # 3. BULLISH ENGULFING
    cond_be = prev_black & is_white & (df['Close'] > lags[1]['Open']) & (df['Open'] < lags[1]['Close']) & (lags[1]['Close'] < lags[7]['Close'])
    add_pattern(cond_be, "Bullish Engulfing")

    # 4. MORNING STAR
    c2_black = lags[2]['Close'] < lags[2]['Open']
    star_setup = c2_black & (lags[2]['Body'] > df['AvgBody']) & (lags[1]['Body'] < (df['AvgBody'] * 0.6))
    cond_ms = star_setup & is_white & (df['Close'] > (lags[2]['Close'] + lags[2]['Body'] * 0.5))
    add_pattern(cond_ms, "Morning Star")

    # 5. MOMENTUM BUY SETUP
    ema_cross_up = df['EMA8'] > df['EMA21']
    momentum_up = df['ROC_5'] > 0
    rsi_healthy = (df['RSI'] > 45) & (df['RSI'] < 65)
    above_vwap = df['Close'] > df['Rolling_VWAP']
    six_month_min_width = df['BandWidth'].rolling(window=126).min()
    catalyst = (df['Vol_Ratio'] > 1.2) | (df['BandWidth'] <= (six_month_min_width * 1.1))
    
    cond_mom = ema_cross_up & momentum_up & rsi_healthy & above_vwap & catalyst
    add_pattern(cond_mom, "Momentum Buy Setup")

    return df

def run_backtest(df, ticker, months_back, fee_bps):
    df = df.copy()
    fee_decimal = fee_bps / 10000.0
    
    # 1. Standard Returns
    df['Ret_D1'] = (df['Close'].shift(-1) / df['Close']) - 1.0 - fee_decimal
    df['Ret_D5'] = (df['Close'].shift(-5) / df['Close']) - 1.0 - fee_decimal
    df['Ret_D15'] = (df['Close'].shift(-15) / df['Close']) - 1.0 - fee_decimal
    
    # 2. VECTORIZED VAULT DOOR STOP-LOSS (European Routing / L&S Spread Accounting)
    # Calculate Spread Multiplier based on liquidity (Volume Ratio)
    conditions = [df['Vol_Ratio'] < 0.8, df['Vol_Ratio'] < 1.0]
    choices = [1.5, 1.2] # 1.5x penalty for severe low volume, 1.2x for mild
    df['Spread_Multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Base buffer: 1.5x ATR * Spread Penalty
    base_risk = df['ATR_14'] * 1.5 * df['Spread_Multiplier']
    
    # Structural Buffer: Flat 40bps to absorb post 17:30 CET algorithmic widening
    extended_hours_buffer = df['Close'] * 0.004
    
    # Calculate maximum allowed drop percentage before ejection
    df['Vault_Door_Price'] = df['Close'] - base_risk - extended_hours_buffer
    df['Vault_Door_Pct'] = (df['Vault_Door_Price'] / df['Close']) - 1.0
    
    # Check if the asset ever breached the Vault Door in the following 15 days
    df['Min_Low_Next_15D'] = df['Low'].rolling(window=15).min().shift(-15)
    df['Max_Drawdown_15D'] = (df['Min_Low_Next_15D'] / df['Close']) - 1.0
    
    # The ultimate test: Did the drawdown exceed our dynamic armor?
    df['Hit_Stop_Loss'] = df['Max_Drawdown_15D'] <= df['Vault_Door_Pct']

    # Filter timeframe
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(months=months_back)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df_recent = df[df.index >= cutoff_date]
    
    # Extract triggers
    trades = df_recent[df_recent['Patterns_Triggered'] != ""].copy()
    
    results = []
    for date, row in trades.iterrows():
        results.append({
            "Date": date.strftime('%Y-%m-%d'),
            "Ticker": ticker,
            "Signals": row['Patterns_Triggered'].strip(" | "),
            "Entry Price": round(row['Close'], 2),
            "Vault Risk Limit": f"{row['Vault_Door_Pct']*100:.2f}%",
            "Hit Stop?": "EJECTED" if row['Hit_Stop_Loss'] else "SURVIVED",
            "D+5 Gain %": round(row['Ret_D5'] * 100, 2) if pd.notna(row['Ret_D5']) else None,
            "D+15 Gain %": round(row['Ret_D15'] * 100, 2) if pd.notna(row['Ret_D15']) else None
        })
    return results

def main():
    print(f"--- VAULT DOOR EXECUTION ENGINE ({BACKTEST_MONTHS} Months) ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Downloading historical data...")
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return
    
    all_results = []
    
    if isinstance(data.columns, pd.MultiIndex):
        if tickers[0] in data.columns.get_level_values(0):
            avail = data.columns.levels[0]
            t_level = 0
        else:
            avail = data.columns.levels[1]
            t_level = 1
    else:
        avail = tickers if len(tickers) == 1 else []
    
    for ticker in avail:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[ticker].copy().dropna() if t_level == 0 else data.xs(ticker, axis=1, level=1).copy().dropna()
            else:
                df_t = data.copy().dropna()

            if len(df_t) < 200: continue
            
            df_t = calculate_technicals_and_patterns(df_t)
            trades = run_backtest(df_t, ticker, BACKTEST_MONTHS, FEE_BPS)
            all_results.extend(trades)
            
        except Exception: 
            continue

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(OUTPUT_FILE_BACKTEST, index=False)
        print(f"\n✅ Backtest Complete. Found {len(all_results)} high-probability setups.")
        print(f"Results saved to {OUTPUT_FILE_BACKTEST}")
        print("\nSample Output (Notice the dynamic risk limit for each trade):")
        print(df_out[['Date', 'Ticker', 'Signals', 'Vault Risk Limit', 'Hit Stop?', 'D+15 Gain %']].tail(10).to_string(index=False))
    else:
        print("No signals found.")

if __name__ == "__main__":
    main()