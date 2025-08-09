# App.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

from Stocks import stock_df, stock_data, peter_lynch, personal_model

st.set_page_config(layout="wide", page_title="Stock Quick View")

# ========= Session flags (persist sections) =========
if "show_details" not in st.session_state:
    st.session_state.show_details = False
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

# ========= Helpers =========
@st.cache_data(ttl=3600)
def load_quick_view(path="quick_view.csv"):
    return pd.read_csv(path)

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

def _clean_series(s: pd.Series) -> pd.Series | None:
    """Make series Plotly‑safe: datetime index, sorted, tz‑naive, float dtype, >=2 pts."""
    if s is None or s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return s

def _try_download_series(ticker: str, period: str, interval: str, auto_adjust: bool):
    """Return a Close series via yf.download (un/maybe adjusted)."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=auto_adjust, progress=False, threads=True)
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    return df["Close"].dropna()

def _try_history_series(ticker: str, period: str, interval: str, auto_adjust: bool):
    """Return a Close series via Ticker.history (un/maybe adjusted)."""
    h = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=auto_adjust)
    if h is None or h.empty:
        return None
    if "Close" not in h.columns:
        return None
    return h["Close"].dropna()

@st.cache_data(ttl=300, show_spinner=False)
def get_series_for_timeframe(ticker, tf_label):
    """
    Robust fetcher:
    1) download(unadjusted) -> history(unadjusted) -> download(adjusted) -> history(adjusted)
    Then slice to requested window when needed.
    """
    def fetch(period, interval):
        s = (_try_download_series(ticker, period, interval, auto_adjust=False)
             or _try_history_series(ticker, period, interval, auto_adjust=False)
             or _try_download_series(ticker, period, interval, auto_adjust=True)
             or _try_history_series(ticker, period, interval, auto_adjust=True))
        return _clean_series(s)

    try:
        if tf_label == "1 Day":
            s = fetch("2d", "5m")
            if s is None:
                return None
            end = s.index[-1]
            start = end - pd.Timedelta("24h")
            return s[(s.index >= start) & (s.index <= end)]

        elif tf_label == "1 Week":
            s = fetch("1mo", "30m")
            if s is None:
                return None
            end = s.index[-1]
            start = end - pd.Timedelta("7d")
            return s[(s.index >= start) & (s.index <= end)]

        elif tf_label == "1 Month":
            return fetch("1mo", "1d")

        elif tf_label == "6 Months":
            return fetch("6mo", "1d")

        elif tf_label == "1 Year":
            return fetch("1y", "1d")

        elif tf_label == "3 Years":
            return fetch("3y", "1wk")

    except Exception:
        return None
    return None

# ========= Header =========
st.title("Stock Quick View")

# ========= Quick View Table =========
st.subheader("Quick View (sortable)")
quick_df = load_quick_view()
disp = quick_df.copy()
num_cols = disp.select_dtypes(include=[np.number]).columns
disp[num_cols] = disp[num_cols].round(2)
disp = disp.fillna("")
st.dataframe(disp, use_container_width=True)

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

            y = s / s.iloc[0] * 100 if normalize else s.astype(float)
            fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name=t))
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
        if empty_list:
            st.caption(f"No data returned for: {', '.join(empty_list[:10])}{'…' if len(empty_list)>10 else ''}")
