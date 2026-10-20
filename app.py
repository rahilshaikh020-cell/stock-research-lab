"""
Stock Research Lab v3 — swing-trading research prototype (India first, USA second).

Educational research tool. NOT investment advice. No signal guarantees profit.

What is new in v3
-----------------
* Prices are adjusted for dividends/splits (Adj Close) before any signal or return is calculated.
* A rebuilt event-based swing backtester:
    - signals use only data up to the day's close; trades execute at the NEXT day's open
    - stop-loss and target are checked against the day's High/Low (intraday), with gap handling
    - if stop and target are both touched in the same bar, the STOP is assumed first (conservative)
    - brokerage/tax cost and slippage are charged on both entry and exit
    - daily equity is rebuilt from real entry/exit prices, then reconciled against the trade log
* In-sample vs out-of-sample reporting, per-stock table, calendar-year table, benchmark comparison.
* Optional risk filters (falling-knife / abnormal price-volume spike) so you can measure whether they help.
* An "Engine audit" tab with hand-checkable self-tests.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TRADING_DAYS = 252


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(symbols, start, end):
    import yfinance as yf

    data = yf.download(
        tickers=symbols, start=start, end=end, auto_adjust=False,
        group_by="ticker", progress=False, threads=True,
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
    """Clean an OHLCV frame and convert O/H/L/C to dividend/split-adjusted values when Adj Close exists."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        low = c.lower().replace("_", " ").strip()
        if low == "adj close":
            rename[c] = "AdjClose"
        elif low in ("open", "high", "low", "close", "volume"):
            rename[c] = low.title()
    df = df.rename(columns=rename)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return None
    numeric = required + (["AdjClose"] if "AdjClose" in df.columns else [])
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    # Adjust OHLC so that returns include dividends and are not broken by splits.
    if "AdjClose" in df.columns:
        factor = (df["AdjClose"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = df[c] * factor
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    return df


def load_csv(uploaded):
    raw = pd.read_csv(uploaded)
    date_col = next((c for c in raw.columns if c.lower().strip() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError("CSV must include a Date column.")
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return normalize_prices(raw)


# ----------------------------------------------------------------------------------------------
# Indicators and signals
# ----------------------------------------------------------------------------------------------
def add_indicators(df):
    d = df.copy().sort_index()
    close, high, low = d["Close"], d["High"], d["Low"]
    d["SMA20"] = close.rolling(20, min_periods=20).mean()
    d["SMA50"] = close.rolling(50, min_periods=50).mean()
    d["SMA200"] = close.rolling(200, min_periods=200).mean()
    d["Return1D"] = close.pct_change()
    d["Volatility20"] = d["Return1D"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    d["VolumeAvg20"] = d["Volume"].rolling(20, min_periods=20).mean()
    # Wilder RSI (same method as most charting platforms)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    d["RSI14"] = 100 - (100 / (1 + rs))
    # Wilder ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    # Highest high of the PRIOR 20 sessions (shifted so today's bar is never included)
    d["High20Prev"] = high.rolling(20, min_periods=20).max().shift(1)
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


SETUPS = ["Trend pullback", "20-day breakout"]


def risk_flags(d):
    """Boolean Series: True where a falling-knife or abnormal price/volume spike condition exists."""
    knife = ((d["Drawdown20High"] <= -0.15) & (d["Close"] < d["SMA50"])) | (
        (d["Close"] < d["SMA50"]) & (d["SMA50"] < d["SMA200"])
    )
    spike = (d["VolumeRatio"] >= 3) & (d["Return1D"].abs() >= 0.08)
    return (knife | spike).fillna(False)


def swing_signal(d, setup, p):
    """
    Entry signal evaluated at the CLOSE of each day (uses only data up to that day).
    The backtester executes it at the NEXT day's open.
    """
    uptrend = (d["SMA50"] > d["SMA200"]) & (d["Close"] > d["SMA200"])
    if setup == "Trend pullback":
        # Uptrend, pulled back (below 20-day average, RSI low), and today closed UP (bounce confirmation).
        sig = uptrend & (d["RSI14"] <= p["rsi_entry"]) & (d["Close"] < d["SMA20"]) & (d["Close"] > d["Close"].shift(1))
    else:
        sig = uptrend & (d["Close"] > d["High20Prev"]) & (d["Close"] > d["SMA50"]) & (d["VolumeRatio"] >= 1.2)
    sig = sig.fillna(False)
    if p.get("risk_filters", True):
        sig = sig & ~risk_flags(d)
    return sig.astype(bool)


# ----------------------------------------------------------------------------------------------
# Swing backtest engine
# ----------------------------------------------------------------------------------------------
def simulate_swing(o, h, l, c, sig, atr, p):
    """
    Event-based long-only simulator for ONE stock. All inputs are numpy arrays of equal length.

    Rules
    -----
    * sig[j-1] True (known at close of day j-1) -> buy at open of day j (plus slippage).
    * stop  = entry open - stop_atr * ATR(j-1);  target = entry open + rr * (entry - stop).
    * Each day while in the trade: time exit at open if held >= max_hold sessions; else gap-down stop at open;
      else intraday stop at the stop price; else gap-up target at open; else intraday target price.
      Stop is tested before target, so a bar that touches both counts as a loss (conservative).
    * Fee (fraction) is charged on entry and on exit; slippage worsens both fills.
    * Daily returns are rebuilt from the real fills so that product(1+daily) == product(trade returns).
    * A trade still open on the last bar is closed at the last close and labelled.
    """
    n = len(o)
    fee, slip = p["fee"], p["slip"]
    trades = []
    ret = np.zeros(n)
    held = np.zeros(n)
    in_pos = False
    i0 = 0
    entry_raw = entry_eff = stop = target = 0.0

    for j in range(1, n):
        if not in_pos:
            if not sig[j - 1]:
                continue
            a = atr[j - 1]
            if not np.isfinite(a) or a <= 0 or not np.isfinite(o[j]):
                continue
            entry_raw = o[j]
            stop = entry_raw - p["stop_atr"] * a
            if stop <= 0:
                continue
            target = entry_raw + p["rr"] * (entry_raw - stop)
            entry_eff = entry_raw * (1 + slip)
            i0 = j
            in_pos = True

        exit_raw, reason = None, None
        if j > i0 and (j - i0) >= p["max_hold"]:
            exit_raw, reason = o[j], "Time exit (open)"
        elif o[j] <= stop:
            exit_raw, reason = o[j], "Stop (gap down at open)"
        elif l[j] <= stop:
            exit_raw, reason = stop, "Stop hit"
        elif o[j] >= target:
            exit_raw, reason = o[j], "Target (gap up at open)"
        elif h[j] >= target:
            exit_raw, reason = target, "Target hit"
        elif j == n - 1:
            exit_raw, reason = c[j], "Open at end of data (marked at close)"

        if exit_raw is None:
            continue

        exit_eff = exit_raw * (1 - slip)
        for k in range(i0, j + 1):
            start_px = entry_eff if k == i0 else c[k - 1]
            end_px = exit_eff if k == j else c[k]
            r = end_px / start_px
            if k == i0:
                r *= 1 - fee
            if k == j:
                r *= 1 - fee
            ret[k] = r - 1
            held[k] = 1
        net = (exit_eff / entry_eff) * (1 - fee) ** 2 - 1
        trades.append({
            "i0": i0, "i1": j, "Entry": entry_raw, "Exit": exit_raw, "Stop": stop, "Target": target,
            "NetReturn": net, "DaysHeld": j - i0, "ExitReason": reason,
            "R": (exit_raw - entry_raw) / (entry_raw - stop),
        })
        in_pos = False

    return trades, ret, held


def run_symbol(df, setup, p):
    d = add_indicators(df)
    valid = d.index[d["SMA200"].notna()]
    if len(valid) < 60:
        return None
    sig = swing_signal(d, setup, p)
    trades, ret, held = simulate_swing(
        d["Open"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float),
        sig.to_numpy(bool), d["ATR14"].to_numpy(float), p,
    )
    idx = d.index
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["EntryDate"] = idx[tdf["i0"].to_numpy()]
        tdf["ExitDate"] = idx[tdf["i1"].to_numpy()]
        tdf = tdf.drop(columns=["i0", "i1"])
    else:
        tdf = pd.DataFrame(columns=["Entry", "Exit", "Stop", "Target", "NetReturn", "DaysHeld", "ExitReason", "R",
                                    "EntryDate", "ExitDate"])
    start = valid[0]
    ret_s = pd.Series(ret, index=idx).loc[start:]
    held_s = pd.Series(held, index=idx).loc[start:]
    bh = d["Close"].pct_change().fillna(0).loc[start:]
    reconciles = True
    if len(tdf):
        compounded_trades = float(np.prod(1 + tdf["NetReturn"].to_numpy()))
        compounded_daily = float(np.prod(1 + ret))
        reconciles = bool(np.isclose(compounded_trades, compounded_daily, rtol=1e-9, atol=1e-12))
    return {"d": d, "signal": sig, "trades": tdf, "ret": ret_s, "held": held_s, "bh": bh, "reconciles": reconciles}


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def trade_stats(tr):
    if tr is None or len(tr) == 0:
        return {"Trades": 0, "Win rate": np.nan, "Profit factor": np.nan, "Expectancy / trade": np.nan,
                "Avg win": np.nan, "Avg loss": np.nan, "Avg days held": np.nan}
    r = tr["NetReturn"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else np.nan
    return {
        "Trades": int(len(r)), "Win rate": float((r > 0).mean()), "Profit factor": float(pf),
        "Expectancy / trade": float(r.mean()),
        "Avg win": float(wins.mean()) if len(wins) else np.nan,
        "Avg loss": float(losses.mean()) if len(losses) else np.nan,
        "Avg days held": float(tr["DaysHeld"].mean()),
    }


def curve_stats(r, held=None):
    if r is None or len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    years = len(r) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 and years > 0 else np.nan
    return {
        "Strategy return": total, "CAGR": cagr,
        "Max drawdown": float((eq / eq.cummax() - 1).min()),
        "Time in market": float(held.mean()) if held is not None else np.nan,
    }


def total_return(r):
    return float((1 + r).prod() - 1) if r is not None and len(r) else np.nan


def segment_report(ret, held, trades, bh, bench, start, end):
    mask = (ret.index >= start) & (ret.index < end)
    r = ret[mask]
    if len(r) < 2:
        return None
    out = curve_stats(r, held[mask])
    if trades is not None and len(trades):
        tr = trades[(trades["EntryDate"] >= r.index[0]) & (trades["EntryDate"] <= r.index[-1])]
    else:
        tr = trades
    out.update(trade_stats(tr))
    out["Buy & hold"] = total_return(bh[mask])
    out["Benchmark"] = total_return(bench[(bench.index >= r.index[0]) & (bench.index <= r.index[-1])]) \
        if bench is not None else np.nan
    return out


def fmt_pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:  # older Streamlit versions
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------------------------
# Engine self-tests (hand-checkable numbers)
# ----------------------------------------------------------------------------------------------
def run_self_tests():
    base = {"stop_atr": 1.0, "rr": 2.0, "max_hold": 10, "fee": 0.0, "slip": 0.0}

    def arr(*v):
        return np.array(v, dtype=float)

    def case(o, h, l, c, sig, p=None, atr_value=2.0):
        p = p or base
        atr = np.full(len(o), atr_value)
        return simulate_swing(arr(*o), arr(*h), arr(*l), arr(*c), np.array(sig, dtype=bool), atr, p)

    results = []

    def check(name, cond, detail):
        results.append({"Test": name, "Result": "PASS" if cond else "FAIL", "Detail": detail})

    # Entry at 100 -> stop 98, target 104.
    t, r, _ = case(o=[100, 100, 101, 100, 100], h=[100, 101, 105, 100, 100], l=[100, 99, 100, 100, 100],
                   c=[100, 100.5, 104, 100, 100], sig=[1, 0, 0, 0, 0])
    check("Target hit intraday -> +4.0%", len(t) == 1 and abs(t[0]["NetReturn"] - 0.04) < 1e-12 and t[0]["Exit"] == 104,
          f"net return {t[0]['NetReturn']:.4f}" if t else "no trade")
    check("Daily returns compound to trade return", len(t) == 1 and abs(np.prod(1 + r) - 1.04) < 1e-12,
          f"compounded {np.prod(1 + r) - 1:.4f}")

    t, _, _ = case(o=[100, 100, 96, 100], h=[100, 101, 97, 100], l=[100, 99, 95, 100], c=[100, 100, 96, 100],
                   sig=[1, 0, 0, 0])
    check("Gap below stop exits at the open (-4.0%), not at the stop", len(t) == 1 and abs(t[0]["NetReturn"] + 0.04) < 1e-12,
          f"net return {t[0]['NetReturn']:.4f}" if t else "no trade")

    t, _, _ = case(o=[100, 100, 100, 100], h=[100, 101, 101, 100], l=[100, 99, 97.5, 100], c=[100, 100, 99, 100],
                   sig=[1, 0, 0, 0])
    check("Intraday stop exits at stop price (-2.0%)", len(t) == 1 and abs(t[0]["NetReturn"] + 0.02) < 1e-12,
          f"net return {t[0]['NetReturn']:.4f}" if t else "no trade")

    t, _, _ = case(o=[100, 100, 100, 100], h=[100, 101, 105, 100], l=[100, 99, 97, 100], c=[100, 100, 100, 100],
                   sig=[1, 0, 0, 0])
    check("Bar touching both stop and target counts as a STOP", len(t) == 1 and t[0]["ExitReason"] == "Stop hit",
          t[0]["ExitReason"] if t else "no trade")

    t, _, _ = case(o=[100, 100, 100, 100], h=[100, 101, 101, 101], l=[100, 99.5, 99.5, 99.5],
                   c=[100, 100, 100, 100], sig=[0, 0, 0, 1])
    check("Signal on the last bar creates no trade (no look-ahead)", len(t) == 0, f"{len(t)} trades")

    t, _, _ = case(o=[100, 100, 100, 100, 100, 100], h=[100, 101, 101, 101, 101, 101],
                   l=[100, 99.5, 99.5, 99.5, 99.5, 99.5], c=[100, 100, 100, 100, 100, 100],
                   sig=[1, 0, 0, 0, 0, 0], p={**base, "max_hold": 2})
    check("Time exit after max_hold sessions, at the open", len(t) == 1 and t[0]["DaysHeld"] == 2 and "Time" in t[0]["ExitReason"],
          f"{t[0]['DaysHeld']} sessions, {t[0]['ExitReason']}" if t else "no trade")

    t, r, _ = case(o=[100, 100, 101, 100], h=[100, 101, 105, 100], l=[100, 99, 100, 100], c=[100, 100.5, 104, 100],
                   sig=[1, 0, 0, 0], p={**base, "fee": 0.001})
    expected = 1.04 * 0.999 ** 2 - 1
    check("Fees charged on both entry and exit", len(t) == 1 and abs(t[0]["NetReturn"] - expected) < 1e-12,
          f"net {t[0]['NetReturn']:.6f} vs expected {expected:.6f}" if t else "no trade")
    check("Daily returns still reconcile after fees", len(t) == 1 and abs(np.prod(1 + r) - (1 + expected)) < 1e-12, "")

    t, _, _ = case(o=[100, 100, 101, 100], h=[100, 101, 105, 100], l=[100, 99, 100, 100], c=[100, 100.5, 104, 100],
                   sig=[1, 0, 0, 0], p={**base, "slip": 0.005})
    expected = (104 * 0.995) / (100 * 1.005) - 1
    check("Slippage worsens both fills", len(t) == 1 and abs(t[0]["NetReturn"] - expected) < 1e-12,
          f"net {t[0]['NetReturn']:.6f} vs expected {expected:.6f}" if t else "no trade")
    return pd.DataFrame(results)


# ----------------------------------------------------------------------------------------------
# Verdict helper
# ----------------------------------------------------------------------------------------------
def build_verdict(oos):
    if not oos:
        return "info", "Not enough out-of-sample history to judge."
    n = oos["Trades"]
    exp, strat, bench = oos["Expectancy / trade"], oos["Strategy return"], oos["Benchmark"]
    if n < 30:
        return "warning", (f"Only {n} out-of-sample trades. That is too few to conclude anything. "
                           "Add more stocks and/or a longer history before trusting any result.")
    if not (exp > 0):
        return "error", ("Out-of-sample expectancy is zero or negative: on unseen data this rule set did not show an edge "
                         "after costs. Do not use it as a trading signal.")
    if pd.notna(bench) and strat <= bench:
        return "warning", ("Out-of-sample trades were profitable on average, but the equal-sleeve strategy did not beat the "
                           "benchmark. Check whether the benefit comes from lower risk (drawdown) or just from luck.")
    return "success", ("Out-of-sample result is positive after costs. That is encouraging but NOT proof: test more stocks, "
                       "other periods, higher costs, and paper-trade before risking money.")


# ----------------------------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")
    st.title("📈 Stock Research Lab v3")
    st.caption("India-first swing-trading research: transparent signals, risk checks, and an audited backtester")
    st.warning(
        "Research prototype only — not investment advice. No signal guarantees profits. "
        "Validate data quality, licensing, costs, and results before any real-money use."
    )

    with st.sidebar:
        st.header("Data")
        market = st.selectbox("Market", ["India", "USA"])
        defaults = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS" if market == "India" else "AAPL,MSFT,NVDA,AMZN,GOOGL"
        symbols_text = st.text_area("Symbols (comma separated)", defaults)
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        today = date.today()
        start_date = st.date_input("Start date", value=today - timedelta(days=365 * 10), max_value=today)
        end_date = st.date_input("End date (exclusive)", value=today, max_value=today)
        uploaded = st.file_uploader("Or upload one stock CSV", type=["csv"])
        st.caption("CSV columns: Date, Open, High, Low, Close, Volume; Adj Close optional.")
        bench_symbol = st.text_input("Benchmark index", "^NSEI" if market == "India" else "^GSPC")

        st.header("Swing strategy")
        setup = st.selectbox("Entry setup", SETUPS)
        rsi_entry = st.slider("Pullback: RSI(14) at or below", 20, 60, 40) if setup == "Trend pullback" else 40
        stop_atr = st.slider("Stop distance (× ATR14)", 1.0, 5.0, 2.0, 0.5)
        rr = st.slider("Target (× risk, reward:risk)", 1.0, 5.0, 2.0, 0.5)
        max_hold = st.slider("Maximum holding (sessions)", 3, 60, 15)
        risk_filters = st.checkbox("Skip falling-knife / abnormal-spike conditions", value=True)

        st.header("Costs and testing")
        fee_bps = st.number_input("Brokerage + taxes per side (bps)", 0.0, 200.0, 15.0, 1.0,
                                  help="Delivery trades in India carry STT, stamp duty, GST and exchange charges. "
                                       "Check your broker's contract note and set this to match.")
        slip_bps = st.number_input("Slippage per side (bps)", 0.0, 200.0, 10.0, 1.0)
        split_pct = st.slider("In-sample share of history (%)", 40, 85, 70)
        st.caption("Tune settings looking at the IN-SAMPLE column only. Every time you adjust a setting after looking at "
                   "the out-of-sample column, that data stops being truly unseen.")

    p = {"rsi_entry": rsi_entry, "stop_atr": stop_atr, "rr": rr, "max_hold": max_hold,
         "risk_filters": risk_filters, "fee": fee_bps / 1e4, "slip": slip_bps / 1e4}

    overview, chart_tab, bt_tab, risk_tab, valuation_tab, audit_tab = st.tabs(
        ["Dashboard", "Stock chart", "Swing backtest", "Risk checks", "Valuation", "Engine audit"]
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

    with audit_tab:
        st.subheader("Backtest engine self-test")
        st.write("These tests feed tiny hand-made price tables to the simulator and check the answers you could calculate "
                 "with a pencil. They test the engine's arithmetic and execution rules, not whether any strategy is profitable.")
        if st.button("Run engine self-tests"):
            tests = run_self_tests()
            show_df(tests)
            if (tests["Result"] == "PASS").all():
                st.success("All engine self-tests passed.")
            else:
                st.error("Some tests failed — do not trust backtest numbers until fixed.")
        st.markdown(
            "**Execution rules used by the backtest**\n"
            "- Signal is known at a day's close; the trade is entered at the **next day's open**.\n"
            "- Prices are dividend/split-adjusted when Adj Close is available.\n"
            "- Stop and target are checked against each day's High/Low; gaps fill at the open; "
            "a bar touching both counts as a **stop**.\n"
            "- Costs and slippage are charged on **both** entry and exit.\n"
            "- Each run reconciles daily equity with the trade log (shown in the Swing backtest tab)."
        )

    if err:
        st.error(err)
    if not prices:
        st.info("Enter symbols in the sidebar or upload a CSV. No synthetic prices are created.")
        return

    enriched = {sym: add_indicators(df) for sym, df in prices.items()}

    # ---- Dashboard -----------------------------------------------------------------------------
    rows = []
    for sym, d in enriched.items():
        if d.empty:
            continue
        last = d.iloc[-1]
        signal, reason = classify(last)
        sw = swing_signal(d, setup, p)
        setup_active = bool(sw.iloc[-1])
        atr = last["ATR14"]
        stop_lvl = last["Close"] - stop_atr * atr if pd.notna(atr) else np.nan
        tgt_lvl = last["Close"] + rr * (last["Close"] - stop_lvl) if pd.notna(stop_lvl) else np.nan
        rows.append({
            "Symbol": sym, "Date": pd.Timestamp(d.index[-1]).date(),
            "Last close": round(float(last["Close"]), 2), "Trend status": signal,
            f"Swing setup ({setup})": "ACTIVE — would enter next open" if setup_active else "—",
            "Ref. stop": round(float(stop_lvl), 2) if setup_active and pd.notna(stop_lvl) else None,
            "Ref. target": round(float(tgt_lvl), 2) if setup_active and pd.notna(tgt_lvl) else None,
            "RSI (14)": round(float(last["RSI14"]), 1) if pd.notna(last["RSI14"]) else None,
            "Close vs SMA50 %": round((last["Close"] / last["SMA50"] - 1) * 100, 2) if pd.notna(last["SMA50"]) else None,
            "Reason": reason,
        })
    scan = pd.DataFrame(rows)
    with overview:
        a, b, c = st.columns(3)
        a.metric("Symbols loaded", len(scan))
        b.metric("Active swing setups", int(scan.filter(like="Swing setup").iloc[:, 0].str.startswith("ACTIVE").sum()) if not scan.empty else 0)
        c.metric("WAIT / risk flags", int((scan["Trend status"] == "WAIT").sum()) if not scan.empty else 0)
        st.subheader("Daily stock scanner")
        show_df(scan)
        st.caption("Reference stop/target use the last close; a real entry happens at the next open, so levels will shift slightly. "
                   "An ACTIVE setup is a rule match, not a recommendation: the backtest tab shows whether the rule has worked historically.")
        st.download_button("Download scanner CSV", scan.to_csv(index=False).encode("utf-8"), "stock_scan.csv", "text/csv")

    # ---- Run backtests ---------------------------------------------------------------------------
    results = {}
    for sym, df in prices.items():
        res = run_symbol(df, setup, p)
        if res is not None:
            results[sym] = res

    # ---- Chart ---------------------------------------------------------------------------------
    with chart_tab:
        sym = st.selectbox("Choose stock", list(enriched.keys()), key="chart_symbol")
        d = enriched[sym].dropna(subset=["Close"])
        fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=sym))
        for col, label in [("SMA20", "SMA 20"), ("SMA50", "SMA 50"), ("SMA200", "SMA 200")]:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=label, mode="lines"))
        if sym in results and len(results[sym]["trades"]):
            tr = results[sym]["trades"]
            fig.add_trace(go.Scatter(x=tr["EntryDate"], y=tr["Entry"], mode="markers", name="Backtest entry",
                                     marker=dict(symbol="triangle-up", size=11, color="green")))
            fig.add_trace(go.Scatter(x=tr["ExitDate"], y=tr["Exit"], mode="markers", name="Backtest exit",
                                     marker=dict(symbol="triangle-down", size=11, color="red")))
        fig.update_layout(title=f"{sym} — daily candlesticks (adjusted prices) with backtest trades", height=600,
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
        signal, reason = classify(d.iloc[-1])
        st.subheader(f"Rule-based trend status: {signal}")
        st.write(reason)

    # ---- Swing backtest ------------------------------------------------------------------------
    with bt_tab:
        st.subheader(f"Swing backtest — {setup}")
        st.write(
            f"Rules: buy next open after the setup appears; stop {stop_atr}× ATR below entry; target {rr}× the risk; "
            f"exit after {max_hold} sessions if neither is hit. Costs {fee_bps:.0f} bps + slippage {slip_bps:.0f} bps per side. "
            f"Risk filters {'ON' if risk_filters else 'OFF'}."
        )
        if not results:
            st.info("Not enough history (need roughly 260+ daily bars per stock) to run the swing backtest.")
        else:
            ret_df = pd.DataFrame({s: r["ret"] for s, r in results.items()})
            held_df = pd.DataFrame({s: r["held"] for s, r in results.items()})
            bh_df = pd.DataFrame({s: r["bh"] for s, r in results.items()})
            port_ret = ret_df.mean(axis=1, skipna=True)
            port_held = held_df.mean(axis=1, skipna=True)
            port_bh = bh_df.mean(axis=1, skipna=True)
            all_trades = pd.concat(
                [r["trades"].assign(Symbol=s) for s, r in results.items() if len(r["trades"])], ignore_index=True
            ) if any(len(r["trades"]) for r in results.values()) else pd.DataFrame(columns=["Symbol", "NetReturn", "EntryDate"])

            bench_ret = None
            if bench_symbol and uploaded is None:
                try:
                    bp = download_prices([bench_symbol], start_date.isoformat(), end_date.isoformat())
                    if bench_symbol in bp:
                        bench_ret = bp[bench_symbol]["Close"].pct_change().dropna()
                except Exception:
                    bench_ret = None
            if bench_ret is None and bench_symbol and uploaded is None:
                st.caption(f"Benchmark {bench_symbol} could not be loaded; benchmark rows are blank.")

            idx = port_ret.index
            split_date = idx[min(int(len(idx) * split_pct / 100), len(idx) - 1)]
            far_future = idx[-1] + pd.Timedelta(days=1)
            segments = {
                f"In-sample (to {split_date.date()})": (idx[0], split_date),
                f"Out-of-sample (from {split_date.date()})": (split_date, far_future),
                "Full period": (idx[0], far_future),
            }
            seg_out = {}
            for name, (s0, s1) in segments.items():
                seg_out[name] = segment_report(port_ret, port_held, all_trades, port_bh, bench_ret, s0, s1)

            order = [
                ("Strategy return (equal sleeves)", "Strategy return", fmt_pct), ("CAGR", "CAGR", fmt_pct),
                ("Max drawdown", "Max drawdown", fmt_pct), ("Time in market", "Time in market", fmt_pct),
                ("Trades", "Trades", lambda x: "0" if x == 0 else f"{int(x)}"),
                ("Win rate", "Win rate", fmt_pct), ("Profit factor", "Profit factor", fmt_num),
                ("Expectancy per trade (net)", "Expectancy / trade", lambda x: fmt_pct(x, 2)),
                ("Average win", "Avg win", fmt_pct), ("Average loss", "Avg loss", fmt_pct),
                ("Average sessions held", "Avg days held", lambda x: fmt_num(x, 1)),
                ("Buy & hold (equal-weight, same stocks)", "Buy & hold", fmt_pct),
                (f"Benchmark ({bench_symbol})", "Benchmark", fmt_pct),
            ]
            table = []
            for label, key, f in order:
                row = {"Metric": label}
                for name, rep in seg_out.items():
                    row[name] = f(rep[key]) if rep and key in rep else "—"
                table.append(row)
            show_df(pd.DataFrame(table))
            st.caption("Equal sleeves: each stock gets an equal slice of capital; a slice sits idle in cash (0% return) while "
                       "its stock has no open trade. Taxes beyond the cost setting, dividends reinvestment details and "
                       "position-size limits are not modelled.")

            level, msg = build_verdict(seg_out[list(segments.keys())[1]])
            getattr(st, level)(msg)

            # equity curves
            eq = (1 + port_ret).cumprod()
            bh_eq = (1 + port_bh).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy (equal sleeves)"))
            fig.add_trace(go.Scatter(x=bh_eq.index, y=bh_eq, name="Buy & hold (equal-weight)"))
            if bench_ret is not None:
                b_eq = (1 + bench_ret[bench_ret.index >= idx[0]]).cumprod()
                fig.add_trace(go.Scatter(x=b_eq.index, y=b_eq, name=f"Benchmark {bench_symbol}"))
            fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(dash="dash", color="gray"))
            fig.add_annotation(x=split_date, y=1, yref="paper", text="out-of-sample starts →", showarrow=False,
                               xanchor="left", yanchor="bottom")
            fig.update_layout(title="Growth of 1 unit of capital", height=420)
            st.plotly_chart(fig)
            dd = (eq / eq.cummax() - 1) * 100
            fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="Drawdown %", fill="tozeroy"))
            fig.update_layout(title="Strategy drawdown (%)", height=260)
            st.plotly_chart(fig)

            # per-stock
            st.subheader("Per-stock results (out-of-sample)")
            stock_rows = []
            for s, r in results.items():
                rep = segment_report(r["ret"], r["held"], r["trades"], r["bh"], None, split_date, far_future)
                if rep:
                    stock_rows.append({
                        "Symbol": s, "OOS trades": rep["Trades"], "OOS strategy return": fmt_pct(rep["Strategy return"]),
                        "OOS buy & hold": fmt_pct(rep["Buy & hold"]), "OOS max drawdown": fmt_pct(rep["Max drawdown"]),
                        "Win rate": fmt_pct(rep["Win rate"]), "Expectancy/trade": fmt_pct(rep["Expectancy / trade"], 2),
                    })
            if stock_rows:
                show_df(pd.DataFrame(stock_rows))
            st.caption("Results for individual stocks are noisy: judge the pooled table above and the number of trades, "
                       "not the best-looking stock.")

            # calendar-year view
            st.subheader("Year-by-year (consistency check)")
            yearly = []
            for yr, grp in port_ret.groupby(port_ret.index.year):
                n_tr = int(((all_trades["EntryDate"].dt.year == yr).sum())) if len(all_trades) else 0
                yearly.append({
                    "Year": int(yr), "Strategy": fmt_pct(total_return(grp)),
                    "Buy & hold": fmt_pct(total_return(port_bh[port_bh.index.year == yr])),
                    "Trades entered": n_tr,
                    "Period": "out-of-sample" if grp.index[-1] >= split_date and grp.index[0] >= split_date else
                              ("mixed" if grp.index[-1] >= split_date else "in-sample"),
                })
            show_df(pd.DataFrame(yearly))

            # audit
            all_ok = all(r["reconciles"] for r in results.values())
            if all_ok:
                st.success("Audit check passed: daily equity compounds exactly to the trade log for every stock.")
            else:
                st.error("Audit check FAILED: daily equity does not reconcile with the trade log. Do not trust these numbers.")

            st.subheader("Trade log")
            tsym = st.selectbox("Stock", ["All stocks"] + list(results.keys()), key="trade_symbol")
            log = all_trades if tsym == "All stocks" else all_trades[all_trades["Symbol"] == tsym]
            if len(log):
                shown = log.copy()
                shown["Net return %"] = (shown["NetReturn"] * 100).round(2)
                shown["Entry date"] = shown["EntryDate"].dt.date
                shown["Exit date"] = shown["ExitDate"].dt.date
                shown = shown[["Symbol", "Entry date", "Entry", "Stop", "Target", "Exit date", "Exit", "Net return %",
                               "R", "DaysHeld", "ExitReason"]].round(2)
                show_df(shown)
                st.download_button("Download trade log", shown.to_csv(index=False).encode("utf-8"), "swing_trade_log.csv", "text/csv")
            else:
                st.info("No trades were generated with these settings.")

    # ---- Risk checks ---------------------------------------------------------------------------
    with risk_tab:
        st.subheader("Falling-knife and abnormal-volume checks")
        for sym, d in enriched.items():
            if d.empty:
                continue
            last = d.iloc[-1]
            signal, reason = classify(last)
            dd = last.get("Drawdown20High", np.nan)
            vr = last.get("VolumeRatio", np.nan)
            r5 = last.get("Return5D", np.nan)
            with st.expander(f"{sym} — {signal}"):
                st.write(reason)
                st.write(f"Change over 5 sessions: {r5 * 100:.2f}%" if pd.notna(r5) else "5-session return unavailable")
                st.write(f"Distance from 20-session high: {dd * 100:.2f}%" if pd.notna(dd) else "20-session high comparison unavailable")
                st.write(f"Volume vs 20-session average: {vr:.2f}x" if pd.notna(vr) else "Volume comparison unavailable")
                st.caption("Anomaly flags are prompts for investigation, not proof of manipulation.")
        st.info("To measure whether these filters actually help, run the Swing backtest twice (filters ON vs OFF) "
                "and compare the out-of-sample column.")

    # ---- Valuation -----------------------------------------------------------------------------
    with valuation_tab:
        st.subheader("Fundamental valuation — manual research inputs")
        st.write("Fundamental data is not automatically sourced in this starter. Enter figures from a reliable filing or "
                 "provider; check the reporting date and currency.")
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
            st.write(f"PEG (rough, P/E divided by growth percentage): {pe / growth:.2f}")
        else:
            st.write("PEG unavailable or not meaningful with current inputs.")
        if fcf_yield > 0:
            st.write("Positive free-cash-flow yield is one potentially useful valuation input; compare within the same industry and verify the source.")
        st.caption("These inputs are not a valuation verdict. Compare peers, business quality, accounting, cyclicality, and data dates.")

    st.divider()
    st.caption("Roadmap: historical constituent universe and delisted stocks; walk-forward re-optimisation; market-specific "
               "tax modelling; reliable fundamentals; portfolio-level position limits; paper trading. Intraday trading is not implemented.")


if __name__ == "__main__":
    main()
