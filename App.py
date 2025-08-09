# App.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

from Stocks import stock_df, stock_data, peter_lynch, personal_model

st.set_page_config(layout="wide", page_title="Stock Quick View")

# ========= Session flags (so sections don't disappear) =========
if "show_details" not in st.session_state:
    st.session_state.show_details = False
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

# ========= Helpers =========
@st.cache_data(ttl=3600)
def load_quick_view(path="quick_view.csv"):
    df = pd.read_csv(path)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)
    return df.fillna("")

def compute_scores(tickers):
    rows = []
    for t in tickers:
        try:
            d = stock_data(t)
            rows.append({
                "Ticker": d.get("ticker", t),
                "Peter Lynch": peter_lynch(d),
                "Personal Model": personal_model(d),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def _last_trading_day_15m(ticker):
    """
    Robust intraday: get 7d/5m, pick the last trading day present,
    return that day's Close series (tz-normalized).
    """
    try:
        df7 = yf.download(ticker, period="7d", interval="5m",
                          auto_adjust=True, progress=False)
        if df7 is None or df7.empty:
            return None
        s = df7["Close"].dropna()
        # Normalize index to UTC; handle tz-naive too
        idx = s.index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        s.index = idx

        # Group by calendar day, pick the last one
        last_day = s.index[-1].date()
        day_mask = s.index.date == last_day
        day_series = s.loc[day_mask]
        return day_series if not day_series.empty else None
    except Exception:
        return None

def get_series_for_timeframe(ticker, tf_label):
    """
    Return a Close-price series for the chosen timeframe.
    """
    try:
        if tf_label == "1 Day":
            # Always take the *last trading day* 5m series
            return _last_trading_day_15m(ticker)

        elif tf_label == "1 Week":
            # Prefer 30m bars, slice last 7 days; fallback to daily
            df = yf.download(ticker, period="1mo", interval="30m",
                             auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                s = df["Close"].dropna()
                idx = s.index
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
                s.index = idx
                cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta("7d")
                s = s[s.index >= cutoff]
                if not s.empty:
                    return s
            # fallback daily
            dfd = yf.download(ticker, period="1mo", interval="1d",
                              auto_adjust=True, progress=False)
            return None if dfd is None or dfd.empty else dfd["Close"].dropna()

        elif tf_label == "1 Month":
            d = yf.download(ticker, period="1mo", interval="1d",
                            auto_adjust=True, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "6 Months":
            d = yf.download(ticker, period="6mo", interval="1d",
                            auto_adjust=True, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "1 Year":
            d = yf.download(ticker, period="1y", interval="1d",
                            auto_adjust=True, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "3 Years":
            d = yf.download(ticker, period="3y", interval="1wk",
                            auto_adjust=True, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

    except Exception:
        return None
    return None

# ========= Header =========
st.title("Stock Quick View")

# ========= Quick View Table =========
st.subheader("Quick View (sortable)")
quick_df = load_quick_view()
st.dataframe(quick_df, use_container_width=True)

st.markdown("---")

# ========= Manual Ticker Lookup =========
st.subheader("Lookup: type tickers")
tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

colA, colB, colC = st.columns(3)
if colA.button("🔍 Get Details"):
    st.session_state.show_details = True
if colB.button("📈 Show Chart"):
    st.session_state.show_chart = True
if colC.button("🧹 Clear Outputs"):
    st.session_state.show_details = False
    st.session_state.show_chart = False

# ========= Details (stock_df + Peter Lynch + Personal Model) =========
if st.session_state.show_details:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        try:
            df_metrics = stock_df(tickers).copy()
            # Ensure Ticker column exists and matches case
            if "Ticker" not in df_metrics.columns:
                df_metrics.rename(columns={"Ticker": "Ticker"}, inplace=True)
            df_metrics["Ticker"] = df_metrics["Ticker"].astype(str).str.upper()

            df_scores = compute_scores(tickers)
            out = df_metrics.merge(df_scores, on="Ticker", how="left") if not df_scores.empty else df_metrics

            num_cols = out.select_dtypes(include=[np.number]).columns
            out[num_cols] = out[num_cols].round(2)

            st.write("**Financial Metrics + Scores**")
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading details: {e}")

# ========= Chart =========
# ---------- REPLACE your chart section with this ----------
if st.session_state.show_chart:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        st.subheader("Price Chart")
        tf = st.selectbox("Timeframe", ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "3 Years"], index=2)
        normalize = st.checkbox("Normalize (start = 100)", value=True, key="norm_chart")

        fig = go.Figure()
        shown, empty_list = 0, []

        for t in tickers:
            s = get_series_for_timeframe(t, tf)
            if s is None or s.empty:
                empty_list.append(t)
                continue
            # make index naive for Plotly if tz is present (keeps hover tidy)
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_convert("UTC").tz_localize(None)

            y = s / s.iloc[0] * 100 if normalize else s
            fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name=t))
            shown += 1

        if shown:
            fig.update_layout(
                height=500,
                template="plotly_white",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Normalized" if normalize else "Price",
                margin=dict(t=30, b=40, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart data to display.")

        # Small debug note so you know which ones had no data for the chosen timeframe
        if empty_list:
            st.caption(f"No data returned for: {', '.join(empty_list[:10])}{'…' if len(empty_list)>10 else ''}")

