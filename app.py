import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")

st.title("📈 Stock Research Lab")
st.caption("India-first stock research, transparent signals, risk checks, and educational backtesting")
st.warning(
    "Research prototype only — not investment advice. No signal guarantees profits. "
    "Validate data quality, licensing, costs, and results before any real-money use."
)

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(symbols, start, end):
    import yfinance as yf
    data = yf.download(
        tickers=symbols, start=start, end=end, auto_adjust=False,
        group_by="ticker", progress=False, threads=True
    )
    result = {}
    if data is None or data.empty:
        return result
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for sym in symbols:
            if sym in available:
                df = normalize_prices(data[sym].copy())
                if df is not None and not df.empty:
                    result[sym] = df
    elif len(symbols) == 1:
        df = normalize_prices(data.copy())
        if df is not None and not df.empty:
            result[symbols[0]] = df
    return result

def normalize_prices(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        low = c.lower().replace("_", " ").strip()
        if low == "adj close": rename[c] = "AdjClose"
        elif low in ("open", "high", "low", "close", "volume"): rename[c] = low.title()
    df = df.rename(columns=rename)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return None
    for c in required + (["AdjClose"] if "AdjClose" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    return df

def load_csv(uploaded):
    raw = pd.read_csv(uploaded)
    date_col = next((c for c in raw.columns if c.lower().strip() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError("CSV must include a Date column.")
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return normalize_prices(raw)

def add_indicators(df):
    d = df.copy().sort_index()
    close = d["Close"]
    d["SMA20"] = close.rolling(20, min_periods=20).mean()
    d["SMA50"] = close.rolling(50, min_periods=50).mean()
    d["SMA200"] = close.rolling(200, min_periods=200).mean()
    d["Return1D"] = close.pct_change()
    d["Volatility20"] = d["Return1D"].rolling(20).std() * np.sqrt(252)
    d["VolumeAvg20"] = d["Volume"].rolling(20, min_periods=20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))
    d["Drawdown20High"] = close / close.rolling(20, min_periods=20).max() - 1
    d["Return5D"] = close.pct_change(5)
    d["Return20D"] = close.pct_change(20)
    d["VolumeRatio"] = d["Volume"] / d["VolumeAvg20"].replace(0, np.nan)
    return d

def classify(last):
    close, sma50, sma200 = last.get("Close"), last.get("SMA50"), last.get("SMA200")
    rsi, dd, vr = last.get("RSI14"), last.get("Drawdown20High"), last.get("VolumeRatio")
    if pd.isna(sma50) or pd.isna(sma200):
        return "WAIT", "At least 200 daily bars are needed for the long-term trend filter."
    if (pd.notna(dd) and dd <= -0.15 and close < sma50) or (close < sma50 and sma50 < sma200):
        return "WAIT", "Falling-knife / weak-trend warning: price and trend filters are weak. A lower price alone is not proof of value."
    if pd.notna(vr) and vr >= 3 and pd.notna(last.get("Return1D")) and abs(last["Return1D"]) >= 0.08:
        return "WAIT", "Abnormal price-volume event: investigate filings, news, liquidity, and possible manipulation. This is not proof of a pump-and-dump."
    if close > sma50 > sma200 and (pd.isna(rsi) or rsi < 70):
        return "BUY CANDIDATE", "Trend conditions pass. Review valuation, fundamentals, liquidity, and risk before any decision."
    if close >= sma200:
        return "HOLD / WATCH", "Long-term trend filter is not bearish, but the entry conditions are incomplete."
    return "WAIT", "Current trend conditions do not qualify as an entry candidate."

def run_backtest(df, cost_bps=10.0, stop_pct=0.08):
    """
    Close-derived signals execute at the next session's open.
    Daily marked-to-market returns use close-to-close returns and prior-session holdings.
    Stop setting is an approximate close-triggered exit executed next open; it is not an intraday stop order.
    """
    d = add_indicators(df).dropna(subset=["SMA50", "SMA200"]).copy()
    if len(d) < 10:
        return d, pd.DataFrame(), {}
    entry_signal = (d["Close"] > d["SMA50"]) & (d["SMA50"] > d["SMA200"]) & (d["RSI14"] < 70)
    exit_signal = (d["Close"] < d["SMA50"]) | (d["RSI14"] > 78)
    dates = d.index
    n = len(d)
    held = np.zeros(n, dtype=float)
    entries, exits = [], []
    state = 0
    entry_price = None
    entry_date = None
    for i in range(n):
        # Position for day i is determined by the prior day's close signal and executes at today's open.
        if i > 0:
            prev = i - 1
            if state == 0 and bool(entry_signal.iloc[prev]):
                state = 1
                entry_price = float(d["Open"].iloc[i])
                entry_date = dates[i]
                entries.append(i)
            elif state == 1:
                close_prev = float(d["Close"].iloc[prev])
                stop_hit = entry_price is not None and close_prev <= entry_price * (1 - stop_pct)
                if bool(exit_signal.iloc[prev]) or stop_hit:
                    state = 0
                    exits.append(i)
                    entry_price = None
                    entry_date = None
        held[i] = state
    # A position held from prior close earns the close-to-close move; execution-day gap precision is approximate.
    close_returns = d["Close"].pct_change().fillna(0)
    previous_held = pd.Series(held, index=dates).shift(1).fillna(0)
    turnover = pd.Series(held, index=dates).diff().abs().fillna(abs(held[0]))
    costs = turnover * (cost_bps / 10000.0)
    strategy_returns = close_returns * previous_held - costs
    equity = (1 + strategy_returns).cumprod()
    buy_hold = (1 + close_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    d["Position"] = held
    d["StrategyReturn"] = strategy_returns
    d["StrategyEquity"] = equity
    d["BuyHoldEquity"] = buy_hold
    d["Drawdown"] = drawdown
    trades = []
    open_trade = None
    for i, idx in enumerate(dates):
        if i in entries:
            open_trade = {"Entry date": idx.date(), "Entry price": float(d["Open"].iloc[i])}
        if i in exits and open_trade:
            px = float(d["Open"].iloc[i])
            ep = open_trade["Entry price"]
            trades.append({**open_trade, "Exit date": idx.date(), "Exit price": px,
                           "Gross return %": round((px / ep - 1) * 100, 2)})
            open_trade = None
    r = strategy_returns
    total = float(equity.iloc[-1] - 1)
    years = max(len(d) / 252, 1 / 252)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    losses = abs(float(r[r < 0].sum()))
    gains = float(r[r > 0].sum())
    metrics = {
        "Total return": total,
        "CAGR (rough)": cagr,
        "Max drawdown": float(drawdown.min()),
        "Buy & hold return": float(buy_hold.iloc[-1] - 1),
        "Positive days": float((r > 0).mean()),
        "Daily-return profit factor": float(gains / losses) if losses > 0 else np.nan,
        "Completed trades": len(trades)
    }
    return d, pd.DataFrame(trades), metrics

with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["India", "USA"])
    defaults = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS" if market == "India" else "AAPL,MSFT,NVDA,AMZN,GOOGL"
    symbols_text = st.text_area("Symbols (comma separated)", defaults)
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=365 * 7), max_value=today)
    end_date = st.date_input("End date (exclusive)", value=today, max_value=today)
    uploaded = st.file_uploader("Or upload one stock CSV", type=["csv"])
    st.caption("CSV columns: Date, Open, High, Low, Close, Volume; Adj Close optional.")
    cost_bps = st.number_input("Estimated cost per position change (bps)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    stop_pct = st.slider("Close-triggered stop threshold (%)", 2, 30, 8) / 100

overview, chart_tab, bt_tab, risk_tab, valuation_tab = st.tabs(
    ["Dashboard", "Stock chart", "Backtest", "Risk checks", "Valuation"]
)

prices, err = {}, None
if uploaded is not None:
    try:
        data = load_csv(uploaded)
        if data is not None and not data.empty:
            prices["UPLOADED"] = data
        else:
            err = "CSV could not be normalized. Check the required OHLCV columns."
    except Exception as exc:
        err = f"CSV error: {exc}"
elif symbols:
    try:
        with st.spinner("Loading daily prices from the selected data source…"):
            prices = download_prices(symbols, start_date.isoformat(), end_date.isoformat())
        if not prices:
            err = "No data returned. Verify symbols/date range or upload a CSV. The free source may be unavailable or incomplete."
    except Exception as exc:
        err = f"Price download failed: {exc}"

if err:
    st.error(err)
if not prices:
    st.info("Enter symbols in the sidebar or upload a CSV. No synthetic prices are created.")
else:
    enriched = {sym: add_indicators(df) for sym, df in prices.items()}
    rows = []
    for sym, d in enriched.items():
        if d.empty: continue
        last = d.iloc[-1]
        signal, reason = classify(last)
        rows.append({
            "Symbol": sym, "Date": pd.Timestamp(d.index[-1]).date(),
            "Last close": round(float(last["Close"]), 4), "Signal": signal,
            "RSI (14)": round(float(last["RSI14"]), 1) if pd.notna(last["RSI14"]) else None,
            "Close vs SMA50 %": round((last["Close"] / last["SMA50"] - 1) * 100, 2) if pd.notna(last["SMA50"]) else None,
            "20-day drawdown %": round(float(last["Drawdown20High"]) * 100, 2) if pd.notna(last["Drawdown20High"]) else None,
            "Reason": reason
        })
    scan = pd.DataFrame(rows)
    with overview:
        a, b, c = st.columns(3)
        a.metric("Symbols loaded", len(scan))
        b.metric("BUY CANDIDATES", int((scan["Signal"] == "BUY CANDIDATE").sum()) if not scan.empty else 0)
        c.metric("WAIT / risk flags", int((scan["Signal"] == "WAIT").sum()) if not scan.empty else 0)
        st.subheader("Daily stock scanner")
        st.dataframe(scan, use_container_width=True, hide_index=True)
        st.download_button("Download scanner CSV", scan.to_csv(index=False).encode("utf-8"), "stock_scan.csv", "text/csv")
    with chart_tab:
        sym = st.selectbox("Choose stock", list(enriched.keys()), key="chart_symbol")
        d = enriched[sym].dropna(subset=["Close"])
        fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=sym))
        for col, label in [("SMA20", "SMA 20"), ("SMA50", "SMA 50"), ("SMA200", "SMA 200")]:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=label, mode="lines"))
        fig.update_layout(title=f"{sym} — daily candlesticks", height=580, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        signal, reason = classify(d.iloc[-1])
        st.subheader(f"Rule-based status: {signal}")
        st.write(reason)
    with bt_tab:
        sym = st.selectbox("Stock to backtest", list(prices.keys()), key="bt_symbol")
        d, trades, m = run_backtest(prices[sym], cost_bps=cost_bps, stop_pct=stop_pct)
        if m:
            cols = st.columns(4)
            cols[0].metric("Strategy return", f"{m['Total return']*100:.2f}%")
            cols[1].metric("CAGR (rough)", f"{m['CAGR (rough)']*100:.2f}%")
            cols[2].metric("Max drawdown", f"{m['Max drawdown']*100:.2f}%")
            cols[3].metric("Buy & hold", f"{m['Buy & hold return']*100:.2f}%")
            st.caption("Research prototype: returns use daily close marks and approximate next-open execution. The stop is close-triggered, not an intraday stop order. Independently validate before relying on these metrics.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.index, y=d["StrategyEquity"], name="Strategy equity"))
            fig.add_trace(go.Scatter(x=d.index, y=d["BuyHoldEquity"], name="Buy & hold"))
            fig.update_layout(title="Growth of 1 unit of capital", height=380)
            st.plotly_chart(fig, use_container_width=True)
            fig = go.Figure(go.Scatter(x=d.index, y=d["Drawdown"] * 100, name="Drawdown %", fill="tozeroy"))
            fig.update_layout(title="Strategy drawdown (%)", height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Additional metrics", m)
            st.subheader("Completed trades")
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button("Download trade log", trades.to_csv(index=False).encode("utf-8"), "trade_log.csv", "text/csv")
        else:
            st.info("Not enough history to calculate this strategy.")
    with risk_tab:
        st.subheader("Falling-knife and abnormal-volume checks")
        for sym, d in enriched.items():
            if d.empty: continue
            last = d.iloc[-1]
            signal, reason = classify(last)
            dd = last.get("Drawdown20High", np.nan)
            vr = last.get("VolumeRatio", np.nan)
            r5 = last.get("Return5D", np.nan)
            with st.expander(f"{sym} — {signal}"):
                st.write(reason)
                st.write(f"Change over 5 sessions: {r5*100:.2f}%" if pd.notna(r5) else "5-session return unavailable")
                st.write(f"Distance from 20-session high: {dd*100:.2f}%" if pd.notna(dd) else "20-session high comparison unavailable")
                st.write(f"Volume vs 20-session average: {vr:.2f}x" if pd.notna(vr) else "Volume comparison unavailable")
                st.caption("Anomaly flags are prompts for investigation, not proof of manipulation.")
    with valuation_tab:
        st.subheader("Fundamental valuation — manual research inputs")
        st.write("Fundamental data is not automatically sourced in this starter. Enter figures from a reliable filing or provider; check the reporting date and currency.")
        sym = st.selectbox("Stock", list(prices.keys()), key="val_symbol")
        col1, col2, col3 = st.columns(3)
        pe = col1.number_input("P/E ratio", min_value=0.0, value=0.0, step=0.5, key="pe")
        pb = col2.number_input("P/B ratio", min_value=0.0, value=0.0, step=0.1, key="pb")
        debt_equity = col3.number_input("Debt/equity", min_value=0.0, value=0.0, step=0.1, key="de")
        col4, col5, col6 = st.columns(3)
        growth = col4.number_input("Earnings growth %", value=0.0, step=1.0, key="growth")
        fcf_yield = col5.number_input("Free-cash-flow yield %", value=0.0, step=0.5, key="fcfy")
        margin = col6.number_input("Operating margin %", value=0.0, step=1.0, key="margin")
        st.write("Inputs for", sym)
        if pe > 0 and growth > 0:
            st.write(f"PEG (rough, P/E divided by growth percentage): {pe/growth:.2f}")
        else:
            st.write("PEG unavailable or not meaningful with current inputs.")
        if fcf_yield > 0:
            st.write("Positive free-cash-flow yield is one potentially useful valuation input; compare within the same industry and verify the source.")
        st.caption("These inputs are not a valuation verdict. Compare peers, business quality, accounting, cyclicality, and data dates.")

st.divider()
st.caption("Roadmap: historical constituent universe and delisted stocks; audited walk-forward/out-of-sample testing; corporate-action handling; market-specific costs/taxes; reliable fundamentals; paper trading. Intraday trading is not implemented.")
