import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "expert_reversal_scan.csv"

def get_trend(series, window=20):
    """Reversals must occur in a downtrend (Price < 20 SMA)."""
    if len(series) < window: return False
    sma = series.rolling(window=window).mean()
    curr = series.iloc[-1]
    return curr < sma.iloc[-1]

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculates Upper and Lower Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def get_company_details(ticker):
    """Fetches news and basic financials."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        financials = {
            "Sector": info.get('sector', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "Target Price": info.get('targetMeanPrice', 'N/A'),
            "Analyst Rec": info.get('recommendationKey', 'N/A').upper() if info.get('recommendationKey') else 'N/A'
        }
        # Latest News
        headlines = []
        if t.news:
            for n in t.news[:2]: # Top 2 news items
                headlines.append(n.get('title', ''))
        financials['News'] = " | ".join(headlines) if headlines else "No News"
        return financials
    except:
        return {}

def check_technical_confluence(df):
    """
    Checks for 'Expert' confirmation signals:
    1. RSI < 30 (Oversold)
    2. Volume Spike (> 2.0x Avg Volume)
    3. Bollinger Band Pierce (Price touched lower band)
    """
    if len(df) < 20: return {}
    
    current = df.iloc[-1]
    
    # 1. RSI
    rsi_series = calculate_rsi(df['Close'])
    current_rsi = rsi_series.iloc[-1]
    rsi_signal = "Oversold" if current_rsi < 30 else "Neutral"
    
    # 2. Volume Spike
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    rel_vol = current['Volume'] / avg_vol if avg_vol > 0 else 0
    vol_signal = "Climax" if rel_vol > 2.0 else "High" if rel_vol > 1.5 else "Normal"
    
    # 3. Bollinger Bands
    _, lower_bb = calculate_bollinger_bands(df['Close'])
    # Did we touch/break the lower band recently?
    bb_dist = (current['Low'] - lower_bb.iloc[-1]) / current['Close']
    bb_signal = "Band Touch" if bb_dist <= 0.005 else "Inside"

    return {
        "RSI": round(current_rsi, 1),
        "RSI Status": rsi_signal,
        "Rel Volume": round(rel_vol, 1),
        "Vol Status": vol_signal,
        "BB Status": bb_signal
    }

def check_patterns_for_ticker(ticker, df):
    if len(df) < 20: return None
    df = df.dropna(subset=['Open', 'Close', 'High', 'Low'])
    
    # Standard Nison Logic
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    
    body0 = abs(c0['Close'] - c0['Open'])
    body1 = abs(c1['Close'] - c1['Open'])
    body2 = abs(c2['Close'] - c2['Open'])
    avg_body = (abs(df['Close'] - df['Open']).tail(20)).mean()
    
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    
    if not get_trend(df['Close']): return None

    patterns = []
    
    # --- Pattern Detection ---
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0): patterns.append("Hammer")
    if (upper0 > 2 * body0) and (lower0 < 0.2 * body0): patterns.append("Inverted Hammer")
    if (c0['Close'] > c0['Open']) and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5): patterns.append("Bullish Belt-Hold")
    if (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']): patterns.append("Bullish Engulfing")
    
    midpoint = c1['Close'] + (body1 / 2)
    if (c1['Close'] < c1['Open'] and body1 > avg_body and c0['Close'] > c0['Open'] and c0['Open'] < c1['Low'] and c0['Close'] > midpoint and c0['Close'] < c1['Open']): patterns.append("Piercing Pattern")
    
    if (c1['Close'] < c1['Open'] and body1 > avg_body and c0['Close'] > c0['Open'] and c0['Close'] < c1['Open'] and c0['Open'] > c1['Close']): patterns.append("Bullish Harami")
    
    if abs(c0['Low'] - c1['Low']) < (c0['Low'] * 0.002): patterns.append("Tweezers Bottom")
    
    if (c2['Close'] < c2['Open'] and body2 > avg_body and abs(c1['Close'] - c1['Open']) < body2 * 0.5 and max(c1['Open'], c1['Close']) < c2['Close'] and c0['Close'] > c0['Open'] and c0['Close'] > (c2['Close'] + body2/2)): patterns.append("Morning Star")
    
    if patterns:
        # Get Expert Confirmations
        techs = check_technical_confluence(df)
        
        result = {
            "Ticker": ticker,
            "Date": df.index[-1].strftime('%Y-%m-%d'),
            "Price": round(c0['Close'], 2),
            "Patterns": ", ".join(patterns),
            "Volume": c0['Volume']
        }
        result.update(techs)
        return result
    return None

def main():
    print("--- EXPERT REVERSAL SCANNER ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col_name = next((c for c in df_tickers.columns if c.lower() == 'ticker'), df_tickers.columns[0])
        tickers = df_tickers[col_name].dropna().astype(str).str.strip().tolist()
    except:
        print("Error: market_tickers.csv not found.")
        return

    print(f"Scanning {len(tickers)} stocks...")
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download error: {e}")
        return
    
    results = []
    
    # Handle single vs multi ticker
    if len(tickers) == 1:
        res = check_patterns_for_ticker(tickers[0], data)
        if res: results.append(res)
    else:
        avail = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else []
        for ticker in avail:
            try:
                df_ticker = data[ticker].copy()
                if df_ticker.empty or df_ticker['Close'].isna().all(): continue
                res = check_patterns_for_ticker(ticker, df_ticker)
                if res:
                    print(f"FOUND: {res['Ticker']} ({res['Patterns']}) - RSI: {res['RSI']}")
                    results.append(res)
            except: continue
    
    if results:
        print("\n--- FETCHING INTELLIGENCE (News & Financials) ---")
        for res in results:
            print(f"Enriching {res['Ticker']}...", end="\r")
            info = get_company_details(res['Ticker'])
            res.update(info)
        
        df_res = pd.DataFrame(results)
        
        # Organize Columns
        cols = ['Ticker', 'Date', 'Price', 'Patterns', 'RSI', 'RSI Status', 'Rel Volume', 'Vol Status', 'BB Status', 'Sector', 'Analyst Rec', 'Target Price', 'News']
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]
        
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\nScan Complete. Results saved to {OUTPUT_FILE}")
        print(df_res[['Ticker', 'Price', 'Patterns', 'RSI Status', 'Vol Status']].to_markdown(index=False))
    else:
        print("No reversals found.")

if __name__ == "__main__":
    main()