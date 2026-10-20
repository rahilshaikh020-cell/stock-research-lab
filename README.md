# Stock Research Lab — beginner starter

This is a starter research website for educational use. It is not investment advice and is not production-ready.

## Run locally (free)

1. Install Python 3.11 or newer from https://www.python.org/downloads/
2. Unzip this folder.
3. Open a terminal / command prompt in this folder.
4. Run:

   ```bash
   python -m venv .venv
   ```

5. Activate the environment:

   **Windows**
   ```bat
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   source .venv/bin/activate
   ```

6. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```
7. Start the website:
   ```bash
   streamlit run app.py
   ```
8. Your browser should open the local app, usually at `http://localhost:8501`.

## Deploy online for free

1. Create a GitHub account.
2. Create a new repository and upload `app.py` and `requirements.txt`.
3. Visit https://share.streamlit.io/ and connect GitHub.
4. Choose the repository and `app.py`, then deploy.
5. Free hosting limits apply. Never upload passwords, API keys, or private account data.

## Use

- India examples: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`
- US examples: `AAPL`, `MSFT`, `NVDA`, `AMZN`
- You can upload a CSV instead. Required columns: `Date, Open, High, Low, Close, Volume`; `Adj Close` is optional.
- Symbols and data availability can change. Verify returned data dates and prices.

## Critical limitations

- The Yahoo Finance download option uses the third-party `yfinance` library. Availability, completeness, and terms can change. Verify the data provider's terms before redistribution or commercial use.
- This app does not yet contain a verified 2,000-stock universe or comprehensive fundamental valuation data.
- Its simple signal rules are illustrative, not validated profitable strategies.
- The backtest is a teaching prototype. Do not use its results for real-money decisions until execution timing, corporate actions, survivorship bias, market-specific fees/taxes, slippage, and out-of-sample methodology have been independently audited.
- Pump-and-dump flags are anomaly alerts only, not proof of manipulation.
- Intraday strategies are not implemented; intraday trading requires intraday data and separate execution models.
