import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")

st.title("📈 Stock Research Lab")
st.caption("Beginner-friendly stock scanner, chart viewer, and educational backtester")
st.warning("Research prototype only — not investment advice. Signals are rule-based hypotheses, not proven predictors. Verify data quality and trading costs before relying on results.")

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(symbols, start, end):
    import yfinance as yf
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True
    )
    result = {}
    if data is None or data.empty:
        return result
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for sym in symbols:
            if sym in available:
                df = data[sym].copy()
                if not df.empty:
                    result[sym] = normalize_prices(df)
    else:
        result[symbols[0]] = normalize_prices(data)
    return {k:v for k,v in result.items() if v is not None and not v.empty}

def normalize_prices(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip().title().replace(" ", "") for c in df.columns]
    # Standardize common Yahoo column names
    rename = {}
    for c in df.columns:
        low = c.lower()
        if low == "adj close": rename[c] = "AdjClose"
        elif low == "open": rename[c] = "Open"
        elif low == "high": rename[c] = "High"
        elif low == "low": rename[c] = "Low"
        elif low == "close": rename[c] = "Close"
        elif low == "volume": rename[c] = "Volume"
    df = df.rename(columns=rename)
    required = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in required):
        return None
    for c in required + (["AdjClose"] if "AdjClose" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"]).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    return df

def load_csv(uploaded):
    raw = pd.read_csv(uploaded)
    # Expected columns: Date, Open, High, Low, Close, Volume; Adj Close optional
    date_col = next((c for c in raw.columns if c.lower() in ("date","datetime","timestamp")), None)
    if not date_col:
        raise ValueError("CSV needs a Date column.")
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return normalize_prices(raw)

def indicators(df):
    d = df.copy()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d["DailyReturn"] = d["Close"].pct_change()
    d["Volatility20"] = d["DailyReturn"].rolling(20).std() * np.sqrt(252)
    d["VolumeAvg20"] = d["Volume"].rolling(20).mean()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))
    d["Return63"] = d["Close"].pct_change(63)
    d["Drawdown20High"] = d["Close"] / d["Close"].rolling(20).max() - 1
    return d

def classify(last):
    close = last.get("Close", np.nan)
    sma50 = last.get("SMA50", np.nan)
    sma200 = last.get("SMA200", np.nan)
    rsi = last.get("RSI14", np.nan)
    dd = last.get("Drawdown20High", np.nan)
    vol = last.get("Volume", np.nan)
    vavg = last.get("VolumeAvg20", np.nan)
    if pd.isna(close) or pd.isna(sma50) or pd.isna(sma200):
        return "WAIT", "Not enough history for 50/200-day trend checks."
    if close < sma50 and sma50 < sma200:
        return "WAIT", "Price is below the 50-day average and the 50-day average is below the 200-day average; possible falling-knife risk."
    if pd.notna(dd) and dd <= -0.15 and close < sma50:
        return "WAIT", "Sharp recent decline with price below the 50-day average; do not treat the price drop alone as a bargain."
    if pd.notna(vavg) and vavg > 0 and vol / vavg >= 3 and pd.notna(dd) and dd <= -0.08:
        return "WAIT", "Unusual volume with a sharp recent reversal/decline; investigate news and liquidity. This does not prove manipulation."
    if close > sma50 and sma50 > sma200 and (pd.isna(rsi) or rsi < 70):
        return "BUY CANDIDATE", "Trend filter is positive. This is a research candidate, not a personalized recommendation."
    if close >= sma200:
        return "HOLD / WATCH", "Price is at or above the 200-day average, but the full entry conditions are not met."
    return "WAIT", "Trend is not strong enough under this simple rule set."

def backtest(df, fee_bps=10, stop_pct=0.08):
    d = indicators(df).dropna(subset=["SMA50","SMA200"]).copy()
    if len(d) < 5:
        return d, pd.DataFrame(), {}
    # Signal is computed at close; position starts on next bar to avoid same-close look-ahead.
    raw_entry = (d["Close"] > d["SMA50"]) & (d["SMA50"] > d["SMA200"]) & (d["RSI14"] < 70)
    raw_exit = (d["Close"] < d["SMA50"]) | (d["RSI14"] > 78)
    position = np.zeros(len(d), dtype=float)
    entries = np.zeros(len(d), dtype=bool)
    exits = np.zeros(len(d), dtype=bool)
    in_pos = False
    entry_price = np.nan
    for i in range(len(d)-1):
        if not in_pos and bool(raw_entry.iloc[i]):
            position[i+1] = 1
            entries[i+1] = True
            in_pos = True
            entry_price = float(d["Open"].iloc[i+1])
        elif in_pos:
            # conservative close-based stop, applied to next open for execution
            stopped = float(d["Close"].iloc[i]) <= entry_price * (1-stop_pct)
            if bool(raw_exit.iloc[i]) or stopped:
                position[i+1] = 0
                exits[i+1] = True
                in_pos = False
            else:
                position[i+1] = 1
    # Carry position state forward; the entry/exit decision uses previous bar's data.
    for i in range(1, len(position)):
        if entries[i]:
            position[i:] = np.maximum(position[i:], 1)
        if exits[i]:
            position[i:] = 0
    # Correct state sequence by rebuilding from entry/exit events
    state = 0
    for i in range(len(position)):
        if entries[i]: state = 1
        if exits[i]: state = 0
        position[i] = state
    daily = d["Close"].pct_change().fillna(0)
    turnover = pd.Series(position, index=d.index).diff().abs().fillna(abs(position[0]))
    costs = turnover * (fee_bps / 10000.0)
    strat = (daily * pd.Series(position, index=d.index).shift(1).fillna(0) - costs)
    equity = (1 + strat).cumprod()
    bh = (1 + daily).cumprod()
    dd = equity / equity.cummax() - 1
    d["StrategyEquity"] = equity
    d["BuyHoldEquity"] = bh
    d["Drawdown"] = dd
    trades = []
    active = False
    ent_date = None
    ent_price = None
    for i, idx in enumerate(d.index):
        if entries[i] and not active:
            active = True
            ent_date = idx
            ent_price = float(d["Open"].iloc[i])
        elif exits[i] and active:
            exit_date = idx
            exit_price = float(d["Open"].iloc[i])
            gross = exit_price / ent_price - 1 if ent_price else np.nan
            trades.append({"Entry date": ent_date.date(), "Exit date": exit_date.date(),
                           "Entry price": round(ent_price, 4), "Exit price": round(exit_price, 4),
                           "Gross return %": round(gross*100, 2)})
            active = False
    rets = strat.dropna()
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    years = max(len(d)/252, 1/252)
    total = float(equity.iloc[-1]-1)
    cagr = float(equity.iloc[-1]**(1/years)-1) if equity.iloc[-1] > 0 else -1
    metrics = {
        "Total return": total,
        "CAGR": cagr,
        "Max drawdown": float(dd.min()),
        "Win days": float((rets > 0).mean()) if len(rets) else 0,
        "Profit factor (daily returns)": float(gains/losses) if losses > 0 else np.nan,
        "Buy & hold return": float(bh.iloc[-1]-1),
        "Trades": len(trades)
    }
    return d, pd.DataFrame(trades), metrics

with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["India", "USA"])
    default_symbols = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS" if market == "India" else "AAPL,MSFT,NVDA,AMZN,GOOGL"
    symbols_text = st.text_area("Symbols (comma separated)", default_symbols, help="Examples: RELIANCE.NS for NSE; AAPL for US. Confirm symbols with your data source.")
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    end_date = date.today()
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365*5), max_value=end_date)
    end_date_input = st.date_input("End date (exclusive)", value=end_date, max_value=end_date)
    uploaded = st.file_uploader("Or upload one stock CSV", type=["csv"])
    st.caption("CSV columns: Date, Open, High, Low, Close, Volume; Adj Close optional.")
    fee_bps = st.number_input("Estimated cost per position change (basis points)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    stop_pct = st.slider("Close-based stop threshold (research setting)", 2, 30, 8) / 100

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Stock chart", "Backtest", "Risk checks"])

price_data = {}
load_error = None
if uploaded is not None:
    try:
        df_upload = load_csv(uploaded)
        price_data["UPLOADED"] = df_upload
    except Exception as e:
        load_error = str(e)
elif symbols:
    try:
        with st.spinner("Loading daily historical prices…"):
            price_data = download_prices(symbols, start_date.isoformat(), end_date_input.isoformat())
        if not price_data:
            load_error = "No price data returned. Check the symbols, dates, internet access, and data-provider availability."
    except Exception as e:
        load_error = f"Could not download prices: {e}. You can upload a CSV instead."

if load_error:
    st.error(load_error)
if not price_data:
    st.info("To begin: enter a few symbols in the sidebar, or upload a CSV with Date, Open, High, Low, Close, Volume columns. The app does not generate fake prices.")
else:
    scan_rows = []
    enriched = {}
    for sym, df in price_data.items():
        d = indicators(df)
        enriched[sym] = d
        if d.empty:
            continue
        last = d.iloc[-1]
        signal, reason = classify(last)
        scan_rows.append({
            "Symbol": sym,
            "Last close": float(last["Close"]),
            "Date": pd.to_datetime(d.index[-1]).date(),
            "Signal": signal,
            "RSI (14)": round(float(last["RSI14"]), 1) if pd.notna(last["RSI14"]) else None,
            "Close vs SMA50 %": round((last["Close"]/last["SMA50"]-1)*100, 2) if pd.notna(last["SMA50"]) else None,
            "Reason": reason
        })
    scan_df = pd.DataFrame(scan_rows)

    with tab1:
        c1,c2,c3 = st.columns(3)
        c1.metric("Symbols loaded", len(scan_df))
        c2.metric("BUY CANDIDATES", int((scan_df["Signal"]=="BUY CANDIDATE").sum()) if not scan_df.empty else 0)
        c3.metric("WAIT alerts", int((scan_df["Signal"]=="WAIT").sum()) if not scan_df.empty else 0)
        st.subheader("Daily stock scanner")
        if not scan_df.empty:
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
            st.download_button("Download scanner CSV", scan_df.to_csv(index=False).encode("utf-8"), "stock_scan.csv", "text/csv")
    with tab2:
        selected = st.selectbox("Choose stock", list(enriched.keys()))
        d = enriched[selected].dropna(subset=["Close"]).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=selected))
        for col, label in [("SMA20","SMA 20"),("SMA50","SMA 50"),("SMA200","SMA 200")]:
            if col in d:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=label))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{selected} — daily candles")
        st.plotly_chart(fig, use_container_width=True)
        last = d.iloc[-1]
        sig, reason = classify(last)
        st.subheader(f"Current rule-based status: {sig}")
        st.write(reason)
        st.caption("This status uses the latest available row from the selected data source. It is not a forecast and may be delayed or adjusted.")
    with tab3:
        selected_bt = st.selectbox("Stock to backtest", list(price_data.keys()), key="btstock")
        d, trades, metrics = backtest(price_data[selected_bt], fee_bps=fee_bps, stop_pct=stop_pct)
        if metrics:
            cols = st.columns(4)
            cols[0].metric("Strategy return", f"{metrics['Total return']*100:.1f}%")
            cols[1].metric("CAGR (rough)", f"{metrics['CAGR']*100:.1f}%")
            cols[2].metric("Max drawdown", f"{metrics['Max drawdown']*100:.1f}%")
            cols[3].metric("Buy & hold", f"{metrics['Buy & hold return']*100:.1f}%")
            st.caption("Illustrative backtest only. This starter engine is not yet a production-grade execution simulator; validate signal timing, costs, corporate actions, and trade logs independently.")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=d.index, y=d["StrategyEquity"], name="Strategy equity"))
            fig2.add_trace(go.Scatter(x=d.index, y=d["BuyHoldEquity"], name="Buy & hold"))
            fig2.update_layout(height=400, title="Growth of 1 unit of capital")
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=d.index, y=d["Drawdown"]*100, name="Drawdown", fill="tozeroy"))
            fig3.update_layout(height=300, title="Strategy drawdown (%)")
            st.plotly_chart(fig3, use_container_width=True)
            st.write("Other metrics", {k:v for k,v in metrics.items() if k not in ("Total return","CAGR","Max drawdown","Buy & hold return")})
            st.subheader("Completed trades")
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button("Download trade log CSV", trades.to_csv(index=False).encode("utf-8"), "trade_log.csv", "text/csv")
        else:
            st.info("Not enough history for this strategy. Load more data.")
    with tab4:
        st.subheader("Risk checks")
        for sym, d in enriched.items():
            if d.empty:
                continue
            last = d.iloc[-1]
            sig, reason = classify(last)
            vol_ratio = last["Volume"] / last["VolumeAvg20"] if pd.notna(last["VolumeAvg20"]) and last["VolumeAvg20"] else np.nan
            drop = last["Drawdown20High"] * 100 if pd.notna(last["Drawdown20High"]) else np.nan
            with st.expander(f"{sym} — {sig}"):
                st.write(reason)
                st.write(f"Distance from 20-day high: {drop:.1f}%" if pd.notna(drop) else "20-day high comparison: not enough data")
                st.write(f"Volume vs 20-day average: {vol_ratio:.2f}x" if pd.notna(vol_ratio) else "Volume comparison: not enough data")
                st.caption("These are simple flags. They cannot establish that a pump-and-dump occurred and do not replace financial-statement, news, or liquidity analysis.")

st.divider()
st.caption("Next steps: verify data licensing; add historical universe membership and delisted stocks; separate long-term, swing, and intraday strategies; implement chronological out-of-sample testing; add Indian and US transaction-cost models; paper-trade before relying on signals.")
