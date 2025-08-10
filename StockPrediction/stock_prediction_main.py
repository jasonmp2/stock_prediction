import yfinance as yf
import pandas as pd
from datetime import datetime
import os

TICKERS = ["MSFT", "SPY", "AAPL", "NVDA", "AMZN", "META", "NFLX", "GOOG", "JPM", "UNH"]
OUTPUT_DIR = r"D:\StockOptionsData"  # <-- Your directory

COLUMNS_TO_KEEP = [
    "contractSymbol", "strike", "lastPrice", "bid", "ask", "change", "percentChange",
    "volume", "openInterest", "impliedVolatility",
    "delta", "gamma", "theta", "vega", "rho",  # Greeks
    "inTheMoney"
]

def fetch_and_save_options_data(ticker: str, save_dir: str):
    stock = yf.Ticker(ticker)
    try:
        current_price = stock.history(period="1d")['Close'].iloc[-1]
    except Exception:
        print(f"Couldn't fetch stock price for {ticker}. Skipping.")
        return

    expirations = stock.options
    data_rows = []

    for expiry in expirations:
        try:
            chain = stock.option_chain(expiry)
            date_collected = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')

            for option_type, df in [('call', chain.calls), ('put', chain.puts)]:
                df = df.copy()
                available_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
                df = df[available_cols]

                df['option_type'] = option_type
                df['expiration'] = expiry
                df['underlying_price'] = current_price
                df['ticker'] = ticker
                df['date_collected'] = date_collected
                df['days_to_expiry'] = (pd.to_datetime(expiry) - pd.to_datetime(date_collected)).days

                data_rows.append(df)

        except Exception as e:
            print(f"Failed to fetch for {ticker} - {expiry}: {e}")

    if data_rows:
        full_df = pd.concat(data_rows, ignore_index=True)
        file_name = f"{ticker}_{timestamp}.csv"
        os.makedirs(save_dir, exist_ok=True)
        full_df.to_csv(os.path.join(save_dir, file_name), index=False)
        print(f"Saved: {file_name} ({len(full_df)} rows)")
    else:
        print(f"No data collected for {ticker}.")

def main():
    for ticker in TICKERS:
        fetch_and_save_options_data(ticker, OUTPUT_DIR)

if __name__ == "__main__":
    main()
