import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from typing import Dict, List
import os

# Define global ticker lists at the top
PIAZZA_AFFARI = [
    "A2A.MI", "AMP.MI", "AZM.MI", "BAMI.MI", "BMED.MI", "BPE.MI", "BZU.MI", "CPR.MI", 
    "DIA.MI", "ENEL.MI", "ENI.MI", "ERG.MI", "RACE.MI", "FBK.MI", "G.MI", "HERA.MI",  # Fixed: HERA.MI
    "INW.MI", "ISP.MI", "IG.MI", "IVG.MI", "LDO.MI", "MB.MI", "MONC.MI", "NEXI.MI", 
    "PIRC.MI", "PRY.MI", "PST.MI", "REC.MI", "RWAY.MI", "SPM.MI", "SRG.MI", "STLAM.MI", 
    "TEN.MI", "TRN.MI", "UCG.MI", "UNI.MI", "US.MI", "WBD.MI", "BC.MI", "IP.MI"
]

NYSE = [
    "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS", "HD", "HON", 
    "IBM", "INTC", "JNJ", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", 
    "CRM", "VZ", "V", "WBA", "WMT", "DIS", "AMZN", "NVDA", "GOOGL", "GOOG", "TSLA", 
    "BRK-B", "META", "MA", "LLY", "AVGO", "ABBV", "PEP", "BAC", "TMO", "ADBE", "NFLX", 
    "CMCSA", "ABT", "ACN", "DHR", "COST", "TXN", "NEE", "RTX", "PM", "UPS", "ORCL", "MS", 
    "BMY", "SCHW", "T", "QCOM", "LOW", "UNP", "SPGI", "INTU", "SBUX", "PLD", "DE", "ISRG", 
    "ELV", "BLK", "MDT", "TJX", "ADI", "SYK", "GILD", "PYPL", "C", "LMT", "AMT", "MDLZ", 
    "CB", "TGT", "MO", "ZTS", "CI", "SO", "DUK", "BDX", "NOC", "CL", "BSX", "EOG", "ITW", 
    "SLB", "FISV", "CSX", "APD", "ICE", "MCK", "WM", "EQIX", "ATVI", "EMR", "NSC", "FDX", 
    "PGR", "SHW", "KLAC", "HCA", "ETN", "NEM", "PSA", "AON", "MCO", "SNPS", "ECL", "ADP", 
    "CDNS", "CME", "APTV", "MCHP", "ORLY", "MAR", "CTAS", "NXPI", "VRTX", "AEP", "FIS", 
    "IDXX", "WELL", "MRNA", "DXCM", "ILMN", "WFC", "MET", "BKNG", "ADSK", "ROP", "LRCX", 
    "GM", "USB", "TFC", "PXD", "AIG", "SPG", "KMB", "SRE", "TDG", "MSCI", "NUE", "DHI", 
    "PSX", "AMP", "FTNT", "KMI", "AFL", "DLR", "WMB", "STZ", "VLO", "CTVA", "MTD", "ODFL", 
    "PCAR", "ANET", "MNST", "WST", "RSG", "PAYX", "YUM", "KEYS", "CPRT", "EFX", "VRSK", 
    "AWK", "TT", "ALGN", "WDC", "SWKS", "TDY", "BIIB", "HUM", "RMD", "EXC", "DDOG", "MTCH", 
    "ZS", "OKTA", "NET", "CRWD", "SNOW", "PLTR", "DASH", "UBER", "LYFT", "ROKU", "SHOP", 
    "MDB", "AFRM", "COIN", "HOOD", "AI", "NIO", "XPEV", "LI", "F", "GE", "LUMN", "PARA", 
    "FOX", "FOXA", "DISH", "EW", "ALC", "REGN", "BIIB", "BNTX", "NVAX", "AZN", "GSK", "SNY", 
    "NVS", "RHHBY"
]

NASDAQ = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST", 
    "PEP", "ADBE", "CSCO", "TMUS", "CMCSA", "AMD", "NFLX", "HON", "INTU", "INTC", "QCOM", 
    "AMGN", "TXN", "SBUX", "GILD", "MDLZ", "VRTX", "ADP", "ISRG", "REGN", "PYPL", "LRCX", 
    "AMAT", "ADI", "PANW", "MU", "CSX", "MRNA", "KLAC", "ORLY", "SNPS", "CDNS", "ASML", 
    "MAR", "CTAS", "NXPI", "ABNB", "FTNT", "CHTR", "MELI", "PDD", "KDP", "KHC", "AEP", 
    "DXCM", "AZN", "ADSK", "BIIB", "IDXX", "WDAY", "ALGN", "CSGP", "SGEN", "VRSK", "FAST", 
    "CRWD", "EXC", "PCAR", "XEL", "PAYX", "DLTR", "CTSH", "ROST", "ODFL", "BKR", "WBA", 
    "EA", "ANSS", "ILMN", "ZS", "SIRI", "DDOG", "CPRT", "MNST", "TTWO", "FANG", "CEG", 
    "TEAM", "VOD", "LCID", "RIVN", "JD", "SWKS", "OKTA", "GFS", "MCHP", "CCEP", "BIDU", 
    "ALNY", "NTES", "WBD", "EBAY", "ZM", "SPLK", "DOCU", "PTON", "ROKU", "SNOW", "PLTR", 
    "UBER", "LYFT", "DASH", "SHOP", "SQ", "COIN", "MDB", "NET", "AFRM", "UPST", "HOOD", 
    "AI", "NIO", "XPEV", "LI", "BYND", "DKNG", "PINS", "SNAP", "SPOT", "CRM", "ORCL", 
    "IBM", "SAP", "VMW", "DELL", "HPQ", "NOK", "ERIC", "T", "VZ", "DIS", "NKE", "MCD", 
    "YUM", "CMG", "LULU", "DISCA", "DISCK", "VIAC", "FOX", "FOXA", "TGT", "HD", "LOW", 
    "BKNG", "EXPE", "ABNB", "HLT", "LVS", "WYNN", "MGM", "BA", "LMT", "NOC", "RTX", "GD", 
    "GE", "CAT", "DE", "MMM", "HON", "UTX", "JNJ", "PFE", "MRK", "ABT", "BMY", "GILD", 
    "AMGN", "REGN", "VRTX", "BIIB", "ILMN", "DXCM", "TMO", "DHR", "BDX", "SYK", "ISRG", 
    "EW", "IDXX", "ZTS", "ALC", "BSX", "MDT", "JNJ", "PFE", "MRNA", "BNTX", "NVAX", "AZN", 
    "GSK", "SNY", "NVS", "RHHBY"
]

# Remove duplicates between NYSE and NASDAQ
NYSE_FINAL = [stock for stock in NYSE if stock not in NASDAQ]

class ComprehensiveBullishScanner:
    def __init__(self):
        self.market_stocks = {
            'piazza_affari': PIAZZA_AFFARI,
            'nyse': NYSE_FINAL, 
            'nasdaq': NASDAQ
        }
        
        print(f"📊 Comprehensive scanner initialized:")
        print(f"   Piazza Affari: {len(PIAZZA_AFFARI)} stocks")
        print(f"   NYSE: {len(NYSE_FINAL)} stocks")
        print(f"   NASDAQ: {len(NASDAQ)} stocks")
        
    def download_data(self, ticker, period='15d'):
        """Download OHLCV data for analysis"""
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in data.columns:
                    return None
            
            data.index = pd.to_datetime(data.index)
            return data
            
        except Exception as e:
            return None

    def try_talib_call(self, func_name, open_prices, high_prices, low_prices, close_prices, **kwargs):
        """Safe TA-Lib function call"""
        fn = getattr(talib, func_name, None)
        if fn is None:
            return np.zeros(len(close_prices), dtype=int)
        try:
            return fn(open_prices, high_prices, low_prices, close_prices, **kwargs)
        except Exception:
            return fn(open_prices, high_prices, low_prices, close_prices)

    def detect_candlestick_patterns(self, df):
        """Detect candlestick patterns using TA-Lib"""
        if len(df) < 5:
            return df

        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values

        # Bullish patterns to monitor
        bullish_patterns = {
            'CDLENGULFING': 'Bullish Engulfing',
            'CDLHAMMER': 'Hammer',
            'CDLINVERTEDHAMMER': 'Inverted Hammer',
            'CDLMORNINGSTAR': 'Morning Star',
            'CDLPIERCING': 'Piercing Pattern',
            'CDL3WHITESOLDIERS': 'Three White Soldiers',
            'CDLMORNINGDOJISTAR': 'Morning Doji Star',
            'CDL3INSIDE': 'Bullish Three Inside',
            'CDL3OUTSIDE': 'Bullish Three Outside',
            'CDLDOJI': 'Doji',
            'CDLDRAGONFLYDOJI': 'Dragonfly Doji'
        }

        for func_name, label in bullish_patterns.items():
            if func_name == 'CDLMORNINGSTAR':
                arr = self.try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices, penetration=0.3)
            else:
                arr = self.try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices)
            
            if len(arr) != len(df):
                arr = np.zeros(len(df), dtype=int)
            
            df[label] = arr

        return df

    def get_last_day_positive_signals(self, df):
        """Extract positive candlestick patterns from the last trading day"""
        if len(df) < 2:
            return []
        
        last_row = df.iloc[-1]
        pattern_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        positive_signals = []
        for pattern in pattern_cols:
            if last_row[pattern] > 0:
                positive_signals.append(pattern)
        
        return positive_signals

    def calculate_performance_metrics(self, df):
        """Calculate 2-week and 1-day performance"""
        if len(df) < 10:
            return None, None, None
        
        # 2-week performance (approx 10 trading days)
        two_week_return = ((df['Close'].iloc[-1] / df['Close'].iloc[-10]) - 1) * 100
        
        # 1-day performance
        daily_return = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
        
        # Current price
        current_price = df['Close'].iloc[-1]
        
        return two_week_return, daily_return, current_price

    def scan_all_stocks(self):
        """Scan all stocks and return comprehensive data"""
        print("🔍 Scanning all stocks for bullish patterns...")
        
        all_stocks_data = []
        
        for market_name, tickers in self.market_stocks.items():
            print(f"   Scanning {market_name.replace('_', ' ').title()}...")
            
            for i, ticker in enumerate(tickers, 1):
                if i % 20 == 0:
                    print(f"      Progress: {i}/{len(tickers)}")
                    
                try:
                    data = self.download_data(ticker, period='15d')
                    if data is None or len(data) < 10:
                        continue
                    
                    # Calculate performance
                    two_week_return, daily_return, current_price = self.calculate_performance_metrics(data)
                    if two_week_return is None:
                        continue
                    
                    # Detect patterns
                    data_with_patterns = self.detect_candlestick_patterns(data)
                    positive_signals = self.get_last_day_positive_signals(data_with_patterns)
                    
                    volume = data['Volume'].iloc[-1]
                    
                    stock_data = {
                        'ticker': ticker,
                        'market': market_name.replace('_', ' ').title(),
                        'signals': positive_signals,
                        'price': current_price,
                        'two_week_return': two_week_return,
                        'daily_return': daily_return,
                        'volume': volume,
                        'signal_count': len(positive_signals),
                        'is_loser': two_week_return < 0  # Flag for 2-week losers
                    }
                    
                    all_stocks_data.append(stock_data)
                        
                except Exception as e:
                    continue
        
        return all_stocks_data

    def generate_comprehensive_report(self):
        """Generate both reports: losers with bullish signals AND top performers overall"""
        print("🚀 COMPREHENSIVE BULLISH PATTERNS REPORT")
        print("=" * 80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()
        
        # Scan all stocks
        all_stocks = self.scan_all_stocks()
        
        if not all_stocks:
            print("❌ No data found for analysis")
            return
        
        # REPORT 1: TOP 10 LOSERS WITH BULLISH SIGNALS (Past 2 weeks)
        print("\n" + "=" * 80)
        print("🏆 TOP 10: LOSERS WITH BULLISH REVERSAL PATTERNS")
        print("=" * 80)
        print("Criteria: Negative 2-week return + Bullish patterns on last trading day")
        print()
        
        # Filter for losers (negative 2-week return) with bullish signals
        losers_with_signals = [stock for stock in all_stocks if stock['is_loser'] and stock['signal_count'] > 0]
        
        if losers_with_signals:
            # Sort by: worst 2-week performance first, then most signals
            losers_with_signals.sort(key=lambda x: (x['two_week_return'], x['signal_count']))
            
            print(f"🎯 Found {len(losers_with_signals)} losers with bullish patterns")
            print("\n🔥 TOP 10 STRONGEST REVERSAL CANDIDATES:")
            print("-" * 70)
            
            for i, candidate in enumerate(losers_with_signals[:10], 1):
                daily_color = "🟢" if candidate['daily_return'] > 0 else "🔴"
                volume_str = f"Vol: {candidate['volume']:,.0f}" if candidate['volume'] > 10000 else "Low Vol"
                
                print(f"{i:2d}. {daily_color} {candidate['ticker']:12} "
                      f"| {candidate['market']:15} | "
                      f"2-wk: {candidate['two_week_return']:+.2f}% | "
                      f"1-day: {candidate['daily_return']:+.2f}% | "
                      f"{candidate['signal_count']:2} signals")
                
                if candidate['signal_count'] > 0:
                    print(f"    Patterns: {', '.join(candidate['signals'][:3])}" + 
                          ("..." if len(candidate['signals']) > 3 else ""))
                print(f"    Price: ${candidate['price']:.2f} | {volume_str}")
                print()
        else:
            print("❌ No stocks found meeting the criteria (2-week losers + bullish patterns)")
        
        # REPORT 2: TOP 10 OVERALL WITH BULLISH SIGNALS
        print("\n" + "=" * 80)
        print("🏆 TOP 10: OVERALL STRONGEST BULLISH SIGNALS")
        print("=" * 80)
        print("Criteria: Most bullish patterns regardless of performance")
        print()
        
        # Filter for stocks with bullish signals
        stocks_with_signals = [stock for stock in all_stocks if stock['signal_count'] > 0]
        
        if stocks_with_signals:
            # Sort by: most signals first, then daily performance
            stocks_with_signals.sort(key=lambda x: (x['signal_count'], x['daily_return']), reverse=True)
            
            print(f"🎯 Found {len(stocks_with_signals)} total stocks with bullish patterns")
            print("\n🔥 TOP 10 STRONGEST BULLISH CANDIDATES:")
            print("-" * 70)
            
            for i, candidate in enumerate(stocks_with_signals[:10], 1):
                daily_color = "🟢" if candidate['daily_return'] > 0 else "🔴"
                two_week_color = "🟢" if candidate['two_week_return'] > 0 else "🔴"
                volume_str = f"Vol: {candidate['volume']:,.0f}" if candidate['volume'] > 10000 else "Low Vol"
                
                print(f"{i:2d}. {daily_color} {candidate['ticker']:12} "
                      f"| {candidate['market']:15} | "
                      f"{candidate['signal_count']:2} signals | "
                      f"1-day: {candidate['daily_return']:+.2f}% | "
                      f"2-wk: {candidate['two_week_return']:+.2f}%")
                
                if candidate['signal_count'] > 0:
                    print(f"    Patterns: {', '.join(candidate['signals'][:3])}" + 
                          ("..." if len(candidate['signals']) > 3 else ""))
                print(f"    Price: ${candidate['price']:.2f} | {volume_str}")
                print()
        else:
            print("❌ No stocks found with bullish patterns")
        
        # SUMMARY STATISTICS
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total stocks analyzed: {len(all_stocks)}")
        print(f"Stocks with bullish patterns: {len(stocks_with_signals)}")
        print(f"2-week losers with bullish patterns: {len(losers_with_signals)}")
        
        if stocks_with_signals:
            avg_signals = sum(s['signal_count'] for s in stocks_with_signals) / len(stocks_with_signals)
            print(f"Average signals per stock: {avg_signals:.1f}")
            
            multi_signal_stocks = [s for s in stocks_with_signals if s['signal_count'] >= 2]
            print(f"Stocks with 2+ signals: {len(multi_signal_stocks)}")

    def quick_overview(self):
        """Quick overview of both categories"""
        print("\n⚡ QUICK OVERVIEW")
        print("-" * 60)
        
        all_stocks = self.scan_all_stocks()
        
        if not all_stocks:
            print("❌ No data found")
            return
        
        # Losers with bullish signals
        losers_with_signals = [s for s in all_stocks if s['is_loser'] and s['signal_count'] > 0]
        losers_with_signals.sort(key=lambda x: (x['two_week_return'], x['signal_count']))
        
        # Overall best signals
        best_signals = [s for s in all_stocks if s['signal_count'] > 0]
        best_signals.sort(key=lambda x: (x['signal_count'], x['daily_return']), reverse=True)
        
        print("🎯 LOSERS WITH BULLISH PATTERNS (Top 5):")
        for stock in losers_with_signals[:5]:
            daily_color = "🟢" if stock['daily_return'] > 0 else "🔴"
            print(f"   {daily_color} {stock['ticker']:12} | "
                  f"2-wk: {stock['two_week_return']:+.2f}% | "
                  f"Signals: {stock['signal_count']} | "
                  f"Price: ${stock['price']:.2f}")
        
        print("\n🎯 STRONGEST BULLISH SIGNALS (Top 5):")
        for stock in best_signals[:5]:
            daily_color = "🟢" if stock['daily_return'] > 0 else "🔴"
            print(f"   {daily_color} {stock['ticker']:12} | "
                  f"Signals: {stock['signal_count']} | "
                  f"1-day: {stock['daily_return']:+.2f}% | "
                  f"Price: ${stock['price']:.2f}")

# Run the comprehensive scanner
if __name__ == "__main__":
    print("🚀 Starting Comprehensive Bullish Patterns Scanner...")
    print("Calculating both: Top 10 losers with bullish signals AND Top 10 overall")
    
    scanner = ComprehensiveBullishScanner()
    
    # Generate comprehensive report with both categories
    scanner.generate_comprehensive_report()
    
    # Quick overview
    scanner.quick_overview()
    
    print("\n✅ Scanning completed! Two reports generated:")
    print("   1. Top 10 losers with bullish reversal patterns")
    print("   2. Top 10 overall strongest bullish signals")