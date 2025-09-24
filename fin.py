import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import mplfinance as mpf

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
   
    # Drop the 'Close' column and use the 'Adj Close'
    #data = data.drop(columns=['Close'])
    # Check if the necessary columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        print(f"Missing columns: {', '.join([col for col in required_columns if col not in data.columns])}")
    data.columns = range(len(data.columns))  
    # Rename columns to desired names
    
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
   
    
    data = data.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'])
    # Keep only required columns and rename 'Adj Close' to 'Close' for mpl finance
    data = data[['Close', 'Open', 'High', 'Low', 'Volume']]

    # Ensure the index is of datetime type
    data.index = pd.to_datetime(data.index)
    
    # Drop rows with missing values in the relevant columns
    data = data.dropna()
    
    return data
    

def detect_candlestick_patterns(df):
    # Drop rows with missing values
    #df = df.dropna(subset=['Open', 'High', 'Low', 'Adj Close'])

    # Check if there are enough rows to calculate patterns
    if len(df) < 5:
        print("Not enough data to detect patterns.")
        return df

    # Convert DataFrame columns to numpy arrays
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values  # Use 'Adj Close' instead of 'Close'
    
    # Ensure all arrays have the same length
    if not (len(open_prices) == len(high_prices) == len(low_prices) == len(close_prices)):
        print("Input arrays have different lengths.")
        return df
    
    # Candlestick pattern detection
    patterns = {
        'Engulfing': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
        'Morning Star': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices, penetration=0.3),
        'Hammer': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
        'Inverted Hammer': talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices),
        'Piercing Pattern': talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices),
        'Three White Soldiers': talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices),
        'Rising Three Methods': talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices, close_prices),
    }

    # Add detected patterns to the DataFrame
    for pattern_name, pattern in patterns.items():
        df[pattern_name] = pattern
    return df


# Filter for selected patterns
def filter_patterns(df):
    detected_patterns = df[(df['Engulfing'] != 0) |
                           (df['Morning Star'] != 0) |
                           (df['Hammer'] != 0) |
                           (df['Inverted Hammer'] != 0) |
                           (df['Piercing Pattern'] != 0) |
                           (df['Three White Soldiers'] != 0) |
                           (df['Rising Three Methods'] != 0)]
    return detected_patterns
    
    

def plot_candlestick_chart(df, ticker):
    # Calculate pivot point, support, and resistance levels
    df['Pivot Point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Support 1'] = (2 * df['Pivot Point']) - df['High']
    df['Resistance 1'] = (2 * df['Pivot Point']) - df['Low']
    # Calculate 8-day EMA
    df['8-Day EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
    # Add pivot points, support/resistance, and 8-EMA
    ap_lines = [
        mpf.make_addplot(df['Pivot Point'], color='black', linestyle='--', width=1.5, label='Pivot Point'),
        mpf.make_addplot(df['Support 1'], color='red', linestyle='--', width=2.5, alpha=0.6, label='Support 1'),
        mpf.make_addplot(df['Resistance 1'], color='green', linestyle='--', width=2.5, alpha=0.6, label='Resistance 1'),
        mpf.make_addplot(df['8-Day EMA'], color='blue', linestyle='-', width=2, label='8-Day EMA')
    ]

    # Plot the candlestick chart
    mpf.plot(
        df,
        type='candle',
        style='charles',
        addplot=ap_lines,
        title=f'{ticker} Candlestick Chart',
        ylabel='Price',
        volume=True,
        figsize=(16, 9) 
    )

tickers = ['AMZN']  
start_date = '2025-08-01'
end_date = '2025-09-23'

for ticker in tickers:
    df = download_data(ticker, start_date, end_date)
    df_with_patterns = detect_candlestick_patterns(df)
    detected_patterns = filter_patterns(df_with_patterns)
    detected_patterns.index = detected_patterns.index.strftime('%Y-%m-%d')
    
    print(f"\nDetected candlestick patterns for {ticker}:\n")
    pattern_columns = detected_patterns.columns[5:]  # Adjusted to the relevant pattern columns
    for idx, row in detected_patterns.iterrows():
        non_zero_patterns = row[pattern_columns][row[pattern_columns] != 0]
        if not non_zero_patterns.empty:
            print(f"Date: {idx}, Patterns: {non_zero_patterns.to_dict()}\n")
    
    # Plot the candlestick chart for each ticker
    plot_candlestick_chart(df, ticker)