# fin.py
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

def download_data(ticker, start_date, end_date):
    """
    Download OHLCV data safely, handling both old/new yfinance formats.
    Forces auto_adjust=False for raw OHLC prices and flattens MultiIndex columns.
    """
    # Force auto_adjust=False to get raw OHLC
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker} between {start_date} and {end_date}")

    # If columns are MultiIndex (new yfinance style), flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure essential columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col not in data.columns:
            data[col] = np.nan

    # Use Adj Close to fill Close if missing
    if data['Close'].isna().all() and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    # Forward fill small gaps
    data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
    data['Volume'] = data['Volume'].fillna(0)

    # Final check: reorder and sanitize
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.index = pd.to_datetime(data.index)
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')

    return data

def try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices, **kwargs):
    """
    Call talib function by name if exists, otherwise return zeros.
    This prevents runtime errors if a TA-Lib build lacks some functions.
    """
    fn = getattr(talib, func_name, None)
    if fn is None:
        # Return zeros (no signal) with same length
        return np.zeros(len(close_prices), dtype=int)
    try:
        return fn(open_prices, high_prices, low_prices, close_prices, **kwargs)
    except TypeError:
        # Some talib functions don't accept kwargs like penetration; call without kwargs
        return fn(open_prices, high_prices, low_prices, close_prices)


def detect_candlestick_patterns(df):
    """
    Add many candlestick pattern columns to df using TA-Lib when available.
    Patterns included follow the common list credited in Steve Nison's literature.
    Returns df with pattern columns (0 = none, positive/negative as TA-Lib returns).
    """

    if len(df) < 5:
        print("Not enough data to detect patterns.")
        return df

    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values

    # List of TA-Lib pattern function names and friendly column names
    talib_patterns = {
        'CDLENGULFING': 'Engulfing',
        'CDLHARAMI': 'Harami',
        'CDLHARAMICROSS': 'Harami Cross',
        'CDLDOJI': 'Doji',
        'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
        'CDLGRAVESTONEDOJI': 'Gravestone Doji',
        'CDLDOJISTAR': 'Doji Star',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLMORNINGDOJISTAR': 'Morning Doji Star' if hasattr(talib, 'CDLMORNINGDOJISTAR') else 'Morning Doji Star',  # may or may not exist
        'CDLPIERCING': 'Piercing Pattern',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDLHAMMER': 'Hammer',
        'CDLHANGINGMAN': 'Hanging Man',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDL3WHITESOLDIERS': 'Three White Soldiers',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDL3INSIDE': 'Three Inside',
        'CDL3OUTSIDE': 'Three Outside',
        'CDLPIERCING': 'Piercing Pattern',
        'CDLONNECK': 'On Neck',
        'CDLINNECK': 'In Neck' if hasattr(talib, 'CDLINNECK') else 'In Neck',
        'CDLTHRUSTING': 'Thrusting',
        'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
        'CDLUPSIDEGAP3METHODS': 'Upside Gap Three Methods',
        'CDLDOWNGAP3METHODS': 'Downside Gap Three Methods' if hasattr(talib, 'CDLDOWNGAP3METHODS') else 'Downside Gap 3 Methods',
        'CDLRISEFALL3METHODS': 'Rising/Falling 3 Methods' if hasattr(talib, 'CDLRISEFALL3METHODS') else 'RiseFall3Methods',
        'CDLSTICKSANDWICH': 'Stick Sandwich' if hasattr(talib, 'CDLSTICKSANDWICH') else 'Stick Sandwich',
        'CDLKICKING': 'Kicking',
        'CDLKICKINGBYLENGTH': 'Kicking by Length' if hasattr(talib, 'CDLKICKINGBYLENGTH') else 'Kicking by Length',
        'CDLMATCHINGLOW': 'Matching Low' if hasattr(talib, 'CDLMATCHINGLOW') else 'Matching Low',
        'CDLCOUNTERATTACK': 'Counterattack' if hasattr(talib, 'CDLCOUNTERATTACK') else 'Counterattack',
        'CDLMARUBOZU': 'Marubozu' if hasattr(talib, 'CDLMARUBOZU') else 'Marubozu',
        'CDLSPINNINGTOP': 'Spinning Top' if hasattr(talib, 'CDLSPINNINGTOP') else 'Spinning Top',
        'CDLSEPARATINGLINES': 'Separating Lines' if hasattr(talib, 'CDLSEPARATINGLINES') else 'Separating Lines',
        'CDLBELTHOLD': 'Belt Hold' if hasattr(talib, 'CDLBELTHOLD') else 'Belt Hold',
        'CDLCOUNTERATTACK': 'Counterattack' if hasattr(talib, 'CDLCOUNTERATTACK') else 'Counterattack',
        'CDLTHREEINSIDE': 'Three Inside'  # duplicate alias protection
    }

    # Deduplicate mapping by function name: keep unique function-name -> label
    unique_funcs = {}
    for fn_name, label in talib_patterns.items():
        unique_funcs[fn_name] = label

    # Compute patterns using TA-Lib if available; if a function doesn't exist, safe fallback
    for func_name, label in unique_funcs.items():
        # Some talib functions accept additional kwargs like penetration; handle special cases
        if func_name == 'CDLMORNINGSTAR':
            arr = try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices, penetration=0.3)
        elif func_name == 'CDLEVENINGSTAR' and hasattr(talib, 'CDLEVENINGSTAR'):
            arr = try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices, penetration=0.3)
        else:
            arr = try_talib_call(func_name, open_prices, high_prices, low_prices, close_prices)

        # Ensure length matches
        if len(arr) != len(df):
            arr = np.zeros(len(df), dtype=int)

        df[label] = arr

    # A few composite pattern detectors that TA-Lib might not have: Tweezers (top/bottom)
    # Tweezers: exact high or low equal-ish across two bars opposite directions
    df['Tweezer Top'] = 0
    df['Tweezer Bottom'] = 0
    if len(df) >= 2:
        # Use a tolerance for equality because exact floats seldom equal
        tol = 1e-8
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            cur = df.iloc[i]
            # Tweezer top: highs approximately equal, prev is bullish and cur is bearish
            if abs(prev['High'] - cur['High']) <= (max(prev['High'], cur['High']) * 1e-4):
                if prev['Close'] > prev['Open'] and cur['Close'] < cur['Open']:
                    df.iloc[i, df.columns.get_loc('Tweezer Top')] = 100
            # Tweezer bottom: lows approximately equal, prev bearish then cur bullish
            if abs(prev['Low'] - cur['Low']) <= (max(prev['Low'], cur['Low']) * 1e-4):
                if prev['Close'] < prev['Open'] and cur['Close'] > cur['Open']:
                    df.iloc[i, df.columns.get_loc('Tweezer Bottom')] = 100

    return df


def filter_patterns(df):
    """
    Return DataFrame rows where any of the pattern columns is non-zero.
    Automatically finds pattern columns (columns not in OHLCV and a few known analytics).
    """
    ignore_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Pivot Point', 'Support 1', 'Resistance 1', '8-Day EMA'}
    # pattern columns are everything not in ignore and not datetime type
    pattern_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype != float]
    # Some pattern columns are integers from TA-Lib; select rows with any non-zero
    cond = np.zeros(len(df), dtype=bool)
    for c in pattern_cols:
        cond = cond | (df[c] != 0)
    return df[cond]


def compute_extrema(df, window=5):
    """
    Identify local minima and maxima using a centered rolling window.
    Returns boolean Series for minima and maxima.
    """
    # Need at least window
    if len(df) < window:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    low_min = df['Low'].rolling(window=window, center=True).min()
    high_max = df['High'].rolling(window=window, center=True).max()

    local_min = (df['Low'] == low_min) & (~low_min.isna())
    local_max = (df['High'] == high_max) & (~high_max.isna())
    return local_min, local_max


def fit_trendline_from_points(index_dates, prices):
    """
    Fit a linear trendline (price vs date) and return trend values across given index_dates.
    index_dates is DatetimeIndex for which we want the trendline values.
    prices correspond to the dates used to fit.
    """
    if len(prices) < 2:
        return None  # not enough points to fit
    # Convert datetimes to numeric (ordinal) for regression
    x = np.array([d.toordinal() for d in index_dates])
    # Fit slope/intercept on the sample dates used for fit (they must be provided similarly)
    # But typical usage: we pass sample_dates and sample_prices; here we will generalize from the
    # sample points. We'll fit a line using polyfit on sample points.
    return None  # placeholder; actual fitting performed in the caller where sample dates available


def plot_candlestick_chart(df, ticker, show_trendlines=True):
    """
    Plot candlestick chart with pivot/support/resistance, EMA, pattern markers, and trendlines.
    """
    df_plot = df.copy()

    # Pivot, support, resistance
    df_plot['Pivot Point'] = (df_plot['High'] + df_plot['Low'] + df_plot['Close']) / 3
    df_plot['Support 1'] = (2 * df_plot['Pivot Point']) - df_plot['High']
    df_plot['Resistance 1'] = (2 * df_plot['Pivot Point']) - df_plot['Low']
    df_plot['8-Day EMA'] = df_plot['Close'].ewm(span=8, adjust=False).mean()

    addplots = [
        mpf.make_addplot(df_plot['Pivot Point'], color='black', linestyle='--', width=1.0),
        mpf.make_addplot(df_plot['Support 1'], color='red', linestyle='--', width=1.0, alpha=0.6),
        mpf.make_addplot(df_plot['Resistance 1'], color='green', linestyle='--', width=1.0, alpha=0.6),
        mpf.make_addplot(df_plot['8-Day EMA'], color='blue', linestyle='-', width=1.5)
    ]

    # Pattern markers: for each row where a pattern exists, put a scatter marker at Close price
    # We'll create a list of series for plotting markers with small vertical offset
    pattern_columns = [c for c in df_plot.columns if c not in {'Open','High','Low','Close','Volume','Pivot Point','Support 1','Resistance 1','8-Day EMA'} and df_plot[c].dtype != float]
    # For clarity, collect up to N pattern series to plot (too many markers may clutter)
    marker_series = {}
    for col in pattern_columns:
        s = df_plot[col].copy()
        s[:] = np.nan
        # set marker at close price where pattern present
        s[df_plot[col] != 0] = df_plot['Close'][df_plot[col] != 0]
        # scale marker upward slightly if it's a bullish pattern (positive in TA-Lib) else downward
        # This is only aesthetic.
        marker_series[col] = s

    # Add up to ~8 pattern marker addplots (mplfinance may handle more but keep reasonable)
    plotted = 0
    for name, ser in marker_series.items():
        if ser.notna().any() and plotted < 12:
            addplots.append(mpf.make_addplot(ser, type='scatter', markersize=80, marker='o', panel=0))
            plotted += 1

    # Trendline estimation: find local minima/maxima and fit linear lines through the last few extremes
    if show_trendlines:
        local_min, local_max = compute_extrema(df_plot, window=5)
        min_idx = df_plot.index[local_min]
        max_idx = df_plot.index[local_max]

        # Choose last N points (most recent extremes) to fit trendlines
        N = 3
        # Support from local minima
        if len(min_idx) >= 2:
            sample_min_idx = min_idx[-N:]
            sample_min_idx = sample_min_idx if len(sample_min_idx) >= 2 else min_idx
            sample_min_prices = df_plot.loc[sample_min_idx, 'Low'].values
            # Fit line
            x = np.array([d.toordinal() for d in sample_min_idx])
            coeffs = np.polyfit(x, sample_min_prices, 1)
            full_x = np.array([d.toordinal() for d in df_plot.index])
            support_line = np.polyval(coeffs, full_x)
            df_plot['Support Trend'] = support_line
            addplots.append(mpf.make_addplot(df_plot['Support Trend'], linestyle='-', width=1.8, alpha=0.9))
        # Resistance from local maxima
        if len(max_idx) >= 2:
            sample_max_idx = max_idx[-N:]
            sample_max_idx = sample_max_idx if len(sample_max_idx) >= 2 else max_idx
            sample_max_prices = df_plot.loc[sample_max_idx, 'High'].values
            x = np.array([d.toordinal() for d in sample_max_idx])
            coeffs = np.polyfit(x, sample_max_prices, 1)
            full_x = np.array([d.toordinal() for d in df_plot.index])
            resistance_line = np.polyval(coeffs, full_x)
            df_plot['Resistance Trend'] = resistance_line
            addplots.append(mpf.make_addplot(df_plot['Resistance Trend'], linestyle='-', width=1.8, alpha=0.9))

    # Final plotting
    mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        addplot=addplots,
        title=f'{ticker} Candlestick Chart',
        ylabel='Price',
        volume=True,
        figsize=(16, 9)
    )


if __name__ == '__main__':
    # Example usage: change tickers, dates as needed
    tickers = ['TSM']
    start_date = '2025-08-01'
    end_date = '2025-10-12'

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = download_data(ticker, start_date, end_date)

        print("Detecting patterns...")
        df_with_patterns = detect_candlestick_patterns(df)

        detected_patterns = filter_patterns(df_with_patterns)
        # Format output: print date and non-zero pattern columns
        if not detected_patterns.empty:
            print(f"\nDetected candlestick patterns for {ticker}:\n")
            for idx, row in detected_patterns.iterrows():
                non_zero = row[[c for c in detected_patterns.columns if c not in {'Open','High','Low','Close','Volume','Pivot Point','Support 1','Resistance 1','8-Day EMA'}]]
                nz = non_zero[non_zero != 0]
                if not nz.empty:
                    # Show pattern names and TA-Lib signals (value sign indicates bull/bear in TA-Lib)
                    print(f"Date: {idx.date()} -> {nz.to_dict()}")
        else:
            print(f"No patterns detected for {ticker} in the period.")

        # Plot chart with trendlines and pattern markers
        plot_candlestick_chart(df_with_patterns, ticker, show_trendlines=True)
