import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "nison_expert_signals_full.csv"

# --- 1. FUNDAMENTAL DATA FETCHER ---
def get_company_details(ticker):
    """Fetches Sector, Analyst Rec, and Target Price."""
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

# --- 2. TECHNICAL INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume Ratio
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['AvgVol']
    
    # Trend & Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['LowerBB'] = df['SMA20'] - (2 * std)
    
    # Candle Body
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    
    return df

def get_trend(row):
    return row['Close'] < row['SMA20']

# --- 3. STRATEGY ENGINE ---
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

# --- 4. PATTERN DETECTION ---
def check_patterns(ticker, df):
    if len(df) < 30: return None
    df = calculate_indicators(df)
    
    c0 = df.iloc[-1] # Today
    c1 = df.iloc[-2] # Yesterday
    c2 = df.iloc[-3] # 2 days ago
    
    if not get_trend(c0): return None
    
    avg_body = c0['AvgBody']
    body0 = c0['Body']
    lower0 = min(c0['Close'], c0['Open']) - c0['Low']
    upper0 = c0['High'] - max(c0['Close'], c0['Open'])
    
    patterns = []
    
    # Definitions
    if (lower0 > 2 * body0) and (upper0 < 0.2 * body0): patterns.append("Hammer")
    if (upper0 > 2 * body0) and (lower0 < 0.2 * body0): patterns.append("Inverted Hammer")
    if (c0['Close'] > c0['Open']) and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5): patterns.append("Bullish Belt-Hold")
    if (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']): patterns.append("Bullish Engulfing")
    if abs(c0['Low'] - c1['Low']) < (c0['Low'] * 0.002): patterns.append("Tweezers Bottom")
    if (c1['Close'] < c1['Open'] and body0 < c1['Body'] and c0['Close'] > c0['Open'] and c0['Close'] < c1['Open'] and c0['Open'] > c1['Close']): patterns.append("Bullish Harami")
    if (c2['Close'] < c2['Open'] and c2['Body'] > avg_body and abs(c1['Close'] - c1['Open']) < c2['Body'] * 0.5 and c0['Close'] > c0['Open'] and c0['Close'] > (c2['Close'] + c2['Body']/2)): patterns.append("Morning Star")

    if patterns:
        primary_pattern = patterns[0]
        rsi = c0['RSI']
        vol = c0['Vol_Ratio']
        
        rating, timeframe, reason = get_trade_advice(primary_pattern, rsi, vol)
        
        # Status Flags
        rsi_status = "Oversold" if rsi < 30 else "Normal"
        vol_status = "Climax" if vol > 2.0 else "High" if vol > 1.5 else "Normal"
        bb_status = "Spring" if c0['Low'] <= c0['LowerBB'] else "Normal"
        
        return {
            "Ticker": ticker,
            "Date": c0.name.strftime('%Y-%m-%d'),
            "Price": round(c0['Close'], 2),
            "Pattern": ", ".join(patterns),
            "Rating": rating,
            "Target Hold": timeframe,
            "Strategy Note": reason,
            
            # Reintegrated Technicals
            "RSI": round(rsi, 1),
            "RSI Status": rsi_status,
            "Rel Volume": round(vol, 1),
            "Vol Status": vol_status,
            "BB Status": bb_status
        }
    return None

def main():
    print("--- NISON EXPERT ADVISOR v2 (Full Data) ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print("Error: market_tickers.csv not found.")
        return

    print(f"Scanning {len(tickers)} stocks...")
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return
        
    results = []
    
    if len(tickers) == 1:
        res = check_patterns(tickers[0], data)
        if res: results.append(res)
    else:
        avail = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else []
        for ticker in avail:
            try:
                df_ticker = data[ticker].copy().dropna()
                res = check_patterns(ticker, df_ticker)
                if res:
                    print(f"FOUND: {ticker} -> {res['Rating']} ({res['Pattern']})")
                    results.append(res)
            except: continue

    if results:
        print(f"\n--- FETCHING COMPANY DETAILS ({len(results)} Candidates) ---")
        # Enhance with Fundamentals
        for res in results:
            print(f"Fetching info for {res['Ticker']}...", end="\r")
            fund_data = get_company_details(res['Ticker'])
            res.update(fund_data)
            
        df_res = pd.DataFrame(results)
        
        # Sort by Rating Priority
        rating_map = {"STRONG BUY": 0, "BUY": 1, "NEUTRAL": 2, "WEAK": 3, "AVOID": 4}
        df_res['Sort_Key'] = df_res['Rating'].map(rating_map)
        df_res = df_res.sort_values('Sort_Key').drop('Sort_Key', axis=1)
        
        # Final Column Order
        cols = [
            'Ticker', 'Date', 'Price', 'Pattern', 'Rating', 'Target Hold', 'Strategy Note',
            'RSI', 'RSI Status', 'Rel Volume', 'Vol Status', 'BB Status',
            'Sector', 'Analyst Rec', 'Target Price'
        ]
        # Filter only cols that exist (safe check)
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]
        
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\nScan Complete. Results saved to {OUTPUT_FILE}")
        print(df_res[['Ticker', 'Price', 'Pattern', 'Rating', 'RSI Status', 'Analyst Rec']].head(10).to_string(index=False))
    else:
        print("No patterns found.")

if __name__ == "__main__":
    main()