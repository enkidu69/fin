import pandas as pd
import yfinance as yf
import numpy as np
import time
import random

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "live_trade_executions.csv"

# --- RISK MANAGEMENT PORTFOLIO CONFIGURATION ---
PORTFOLIO_SIZE_EUR = 10000.00  # Set your actual Trade Republic account balance here
KELLY_FRACTION = 0.5           # 0.5 = Half-Kelly (Professional Standard)
EMPIRICAL_WIN_RATE = 0.4645    # From your quantitative backtest
EMPIRICAL_RR_RATIO = 1.61      # From your quantitative backtest

# --- 1. ROBUST DOWNLOADER ---
def safe_download(ticker, period="2y", interval="1d"):
    try:
        time.sleep(random.uniform(0.1, 0.5)) 
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if interval != "1d" and not df.empty: 
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if c[0] != 'Datetime' else 'Datetime' for c in df.columns]
                close_col = [c for c in df.columns if 'Close' in str(c)]
                if close_col: df['Close'] = df[close_col[0]]
            df = df.set_index('Datetime')
        return df
    except:
        return pd.DataFrame()

# --- 2. TECHNICAL & PATTERN ENGINE ---
def calculate_technicals_and_patterns(df):
    df = df.copy()
    
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

    # Volatility for the Vault Door
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # Candlestick Anatomy
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

    # Core Nison Reversals
    cond_h10 = (df['LowerShadow'] > 2 * df['Body']) & (df['UpperShadow'] < 0.2 * df['Body']) & (lags[1]['Close'] < lags[10]['Close'])
    add_pattern(cond_h10, "Hammer 10")

    cond_ih = (lags[1]['UpperShadow'] > 2 * lags[1]['Body']) & (lags[1]['LowerShadow'] < 0.2 * lags[1]['Body']) & (df['Close'] < lags[3]['Close']) & (df['Close'] > lags[1]['Close'])
    add_pattern(cond_ih, "Inverted Hammer")

    cond_be = prev_black & is_white & (df['Close'] > lags[1]['Open']) & (df['Open'] < lags[1]['Close']) & (lags[1]['Close'] < lags[7]['Close'])
    add_pattern(cond_be, "Bullish Engulfing")

    c2_black = lags[2]['Close'] < lags[2]['Open']
    star_setup = c2_black & (lags[2]['Body'] > df['AvgBody']) & (lags[1]['Body'] < (df['AvgBody'] * 0.6))
    cond_ms = star_setup & is_white & (df['Close'] > (lags[2]['Close'] + lags[2]['Body'] * 0.5))
    add_pattern(cond_ms, "Morning Star")

    ema_cross_up = df['EMA8'] > df['EMA21']
    momentum_up = df['ROC_5'] > 0
    rsi_healthy = (df['RSI'] > 45) & (df['RSI'] < 65)
    above_vwap = df['Close'] > df['Rolling_VWAP']
    six_month_min_width = df['BandWidth'].rolling(window=126).min()
    catalyst = (df['Vol_Ratio'] > 1.2) | (df['BandWidth'] <= (six_month_min_width * 1.1))
    
    cond_mom = ema_cross_up & momentum_up & rsi_healthy & above_vwap & catalyst
    add_pattern(cond_mom, "Momentum Buy")

    return df

# --- 3. RISK MANAGEMENT MODULE ---
def calculate_vault_door(current_price, atr, vol_ratio):
    """Calculates the L&S adjusted Stop Loss."""
    if pd.isna(atr) or atr == 0: return current_price * 0.97, -0.03 # Fallback
    
    # Base Volatility Buffer
    spread_multiplier = 1.5 if vol_ratio < 0.8 else (1.2 if vol_ratio < 1.0 else 1.0)
    base_risk = atr * 1.5 * spread_multiplier
    
    # Lang & Schwarz Extended Hours Tax Buffer (40 bps)
    ls_spread_tax = current_price * 0.004
    
    vault_door_price = current_price - base_risk - ls_spread_tax
    risk_pct = (vault_door_price / current_price) - 1.0
    
    # Hard bounds: Never risk more than 7% mathematically, never less than 1.5%
    if risk_pct < -0.07: vault_door_price = current_price * 0.93
    if risk_pct > -0.015: vault_door_price = current_price * 0.985
    
    final_pct = (vault_door_price / current_price) - 1.0
    return vault_door_price, final_pct

def calculate_kelly_allocation(portfolio_size):
    """Calculates exactly how much Euro to risk based on empirical edge."""
    full_kelly = EMPIRICAL_WIN_RATE - ((1 - EMPIRICAL_WIN_RATE) / EMPIRICAL_RR_RATIO)
    target_kelly = full_kelly * KELLY_FRACTION # Smooths out drawdowns
    
    allocation = portfolio_size * target_kelly
    return allocation, target_kelly

# --- 4. MAIN EXECUTION ENGINE ---
def main():
    print(f"--- TRADE REPUBLIC / L&S EXECUTION ENGINE ---")
    print(f"Portfolio Size: €{PORTFOLIO_SIZE_EUR:,.2f}")
    
    target_alloc_eur, kelly_pct = calculate_kelly_allocation(PORTFOLIO_SIZE_EUR)
    print(f"Empirical Half-Kelly Allocation: €{target_alloc_eur:,.2f} ({kelly_pct*100:.2f}% per trade)\n")
    print("Scanning market for Tier-1 Reversals & Momentum...\n")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print(f"Error: {INPUT_FILE} not found.")
        return

    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True, progress=False)
    except Exception as e:
        print(f"Download Error: {e}")
        return
    
    active_trades = []
    
    if isinstance(data.columns, pd.MultiIndex):
        t_level = 0 if tickers[0] in data.columns.get_level_values(0) else 1
        avail = data.columns.levels[t_level]
    else:
        avail = tickers if len(tickers) == 1 else []
        t_level = -1
    
    for ticker in avail:
        try:
            if t_level == 0: df_t = data[ticker].copy().dropna()
            elif t_level == 1: df_t = data.xs(ticker, axis=1, level=1).copy().dropna()
            else: df_t = data.copy().dropna()

            if len(df_t) < 30: continue
            
            df_t = calculate_technicals_and_patterns(df_t)
            c0 = df_t.iloc[-1]
            
            signals = c0['Patterns_Triggered'].strip(" | ")
            if signals:
                price = c0['Close']
                atr = c0['ATR_14']
                vol = c0['Vol_Ratio']
                
                vault_price, vault_pct = calculate_vault_door(price, atr, vol)
                
                # Boost allocation slightly if multiple independent signals trigger
                signal_count = len(signals.split(" | "))
                final_alloc = target_alloc_eur * (1.2 if signal_count > 1 else 1.0)
                
                active_trades.append({
                    "Date": c0.name.strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Setup Triggered": signals,
                    "Entry Price": round(price, 2),
                    "Vault Door SL": f"€{vault_price:.2f} ({vault_pct*100:.2f}%)",
                    "Kelly Allocation": f"€{final_alloc:.2f}",
                    "Vol Fuel": round(vol, 2)
                })
        except Exception: 
            continue

    if active_trades:

        df_out = pd.DataFrame(active_trades)
        
        # --- NEW: CONVICTION SCORING SYSTEM ---
        # 1. Base Score = Volume Ratio (Fuel to beat the spread tax)
        df_out['Conviction_Score'] = df_out['Vol Fuel'] * 10 
        
        # 2. Add points for high-EV patterns
        df_out.loc[df_out['Setup Triggered'].str.contains('Hammer 10'), 'Conviction_Score'] += 5
        df_out.loc[df_out['Setup Triggered'].str.contains('Morning Star'), 'Conviction_Score'] += 3
        
        # 3. Add points for signal confluence (Multiple patterns triggering together)
        df_out['Signal_Count'] = df_out['Setup Triggered'].str.count(r'\|') + 1
        df_out['Conviction_Score'] += (df_out['Signal_Count'] * 2)
        
        # 4. Deduct points for high Vault Door risk (Extract the % number and penalize)
        # e.g., A -5.00% risk deducts 5 points. A -2.00% risk deducts only 2 points.
        df_out['Risk_Penalty'] = df_out['Vault Door SL'].str.extract(r'\(([-0-9.]+)%\)').astype(float).abs()
        df_out['Conviction_Score'] -= df_out['Risk_Penalty']
        
        # Sort by the highest score to find the #1 daily trade
        # Sort by the highest score to find the #1 daily trade
        df_out = df_out.sort_values(by='Conviction_Score', ascending=False).reset_index(drop=True)
        
        # --- NEW: DATED OUTPUT FILE ---
        # Generate a dynamic filename using today's date
        today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
        dated_filename = f"live_trade_executions_{today_str}.csv"
        
        # Save ALL identified setups to the dated file (dropping the internal scoring columns for clean reading)
        df_out.drop(columns=['Conviction_Score', 'Signal_Count', 'Risk_Penalty']).to_csv(dated_filename, index=False)
        
        print(f"✅ FOUND {len(active_trades)} EXECUTABLE SETUPS TODAY.")
        print(f"📁 Full log of all {len(active_trades)} setups saved to: {dated_filename}")
        
        print("\n--- 🏆 THE #1 TOP RANKED TRADE FOR TODAY ---")
        top_trade = df_out.iloc[[0]].drop(columns=['Conviction_Score', 'Signal_Count', 'Risk_Penalty'])
        print(top_trade.to_string(index=False))
        
        if len(df_out) > 1:
            print("\n--- ALTERNATE SETUPS (DO NOT EXECUTE IF TOP TRADE IS TAKEN) ---")
            runner_ups = df_out.iloc[1:4].drop(columns=['Conviction_Score', 'Signal_Count', 'Risk_Penalty'])
            print(runner_ups.to_string(index=False))
            
        print("\nACTION PLAN: Execute the #1 trade. Set your Stop-Loss at the Vault Door price immediately.")
        
        print(f"✅ FOUND {len(active_trades)} EXECUTABLE SETUPS TODAY.")
        print("\n--- 🏆 THE #1 TOP RANKED TRADE FOR TODAY ---")
        top_trade = df_out.iloc[[0]].drop(columns=['Conviction_Score', 'Signal_Count', 'Risk_Penalty'])
        print(top_trade.to_string(index=False))
        
        if len(df_out) > 1:
            print("\n--- ALTERNATE SETUPS (DO NOT EXECUTE IF TOP TRADE IS TAKEN) ---")
            runner_ups = df_out.iloc[1:4].drop(columns=['Conviction_Score', 'Signal_Count', 'Risk_Penalty'])
            print(runner_ups.to_string(index=False))
            
        print("\nACTION PLAN: Execute the #1 trade. Set your Stop-Loss at the Vault Door price immediately.")
    else:
        print("Market is quiet. No tier-one signals triggered today. Protect your capital and wait.")

if __name__ == "__main__":
    main()