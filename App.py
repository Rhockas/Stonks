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

# ========= Quick table loader =========
@st.cache_data(ttl=3600)
def load_quick_view(path="quick_view.csv"):
    return pd.read_csv(path)

# ========= Score helper for details =========
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

# ========= CHART HELPERS (robust + simple fallbacks) =========
def _clean_series(s: pd.Series) -> pd.Series | None:
    if s is None or len(s) == 0:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s if len(s) > 1 else None

def _slice_last(s: pd.Series, delta: str) -> pd.Series | None:
    if s is None or s.empty:
        return None
    end = s.index[-1]
    start = end - pd.Timedelta(delta)
    return _clean_series(s[(s.index >= start) & (s.index <= end)])

def _try_history(ticker: str, period: str, interval: str):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty and "Close" in df:
            return df["Close"].dropna()
    except Exception:
        pass
    return None

def _try_download(ticker: str, period: str, interval: str):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is not None and not df.empty and "Close" in df:
            return df["Close"].dropna()
    except Exception:
        pass
    return None

def fetch_series_for_chart(ticker: str, period_label: str) -> pd.Series | None:
    """
    Robust chain with fallbacks:
      1D  : 2d/5m (history) -> 2d/5m (download) -> 2d/1d sliced to 24h
      1W  : 14d/30m -> 30d/60m -> 10d/1d
      1M  : 60d/60m (resample 4H) -> 1mo/1d
      6M  : 6mo/1d
      1Y  : 1y/1d
      3Y  : 3y/1wk
    """
    try:
        if period_label == "1 Day":
            s = _try_history(ticker, "2d", "5m") or _try_download(ticker, "2d", "5m")
            if s is None or s.empty:
                s = _try_download(ticker, "2d", "1d")
                return _slice_last(s, "24h") if s is not None else None
            return _slice_last(s, "24h")

        if period_label == "1 Week":
            s = _try_download(ticker, "14d", "30m")
            if s is None or s.empty:
                s = _try_download(ticker, "30d", "60m")
            if s is None or s.empty:
                s = _try_download(ticker, "10d", "1d")
                return _slice_last(s, "7d") if s is not None else None
            return _slice_last(s, "7d")

        if period_label == "1 Month":
            s = _try_download(ticker, "60d", "60m")
            if s is not None and not s.empty:
                s = _slice_last(s, "30d")
                if s is not None and not s.empty:
                    s4h = s.resample("4H").last().dropna()
                    return _clean_series(s4h)
            # fallback daily
            s = _try_download(ticker, "1mo", "1d")
            return _clean_series(s)

        if period_label == "6 Months":
            s = _try_download(ticker, "6mo", "1d")
            return _clean_series(s)

        if period_label == "1 Year":
            s = _try_download(ticker, "1y", "1d")
            return _clean_series(s)

        if period_label == "3 Years":
            s = _try_download(ticker, "3y", "1wk")
            return _clean_series(s)
    except Exception:
        return None
    return None

# Simple-mode (matches your original approach) if robust path yields nothing.
SIMPLE_MAP = {
    "1 Day":   ("1d",  "5m"),
    "1 Week":  ("5d",  "30m"),
    "1 Month": ("1mo", "1d"),
    "6 Months":("6mo", "1d"),
    "1 Year":  ("1y",  "1d"),
    "3 Years": ("3y",  "1wk"),
}

def simple_fetch_series(ticker: str, period_label: str) -> pd.Series | None:
    period, interval = SIMPLE_MAP[period_label]
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is not None and not df.empty and "Close" in df:
            return _clean_series(df["Close"].dropna())
    except Exception:
        pass
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

# ========= Chart (same UI; robust + simple fallback) =========
if st.session_state.show_chart:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        st.subheader("Price Chart")
        period_label = st.selectbox(
            "Select time period:",
            ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "3 Years"],
            index=2
        )
        use_normalized = st.checkbox("Normalize prices (start from 100)", value=True)

        price_data = {}
        empties = []

        for t in tickers:
            s = fetch_series_for_chart(t, period_label)
            if s is None or s.empty:
                # try the simple/original path as a fallback
                s = simple_fetch_series(t, period_label)
            if s is not None and not s.empty:
                price_data[t] = s
            else:
                empties.append(t)

        selected = st.multiselect(
            "Select tickers to show in chart:",
            options=list(price_data.keys()),
            default=list(price_data.keys())
        )

        if selected:
            fig = go.Figure()
            y_min, y_max = None, None

            for t in selected:
                series = price_data[t]
                y_vals = series.to_numpy(dtype="float64")
                if len(y_vals) < 2:
                    continue

                if use_normalized:
                    y_vals = (y_vals / y_vals[0]) * 100.0

                ymin = float(np.min(y_vals))
                ymax = float(np.max(y_vals))
                y_min = ymin if y_min is None else min(y_min, ymin)
                y_max = ymax if y_max is None else max(y_max, ymax)

                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=y_vals,
                    mode='lines',
                    name=t,
                    hovertemplate = (
                        f"{t}<br>"
                        "Date: %{x|%Y-%m-%d %H:%M}<br>"
                        + ("Norm " if use_normalized else "")
                        + "Price: %{y:.2f}<extra></extra>"
                    )
                ))

            tick_vals = None
            if y_min is not None and y_max is not None:
                y_range = y_max - y_min
                if y_range > 0:
                    tick_spacing = max(round(y_range / 10), 1)
                    tick_start = int(y_min // tick_spacing * tick_spacing)
                    tick_end = int(y_max // tick_spacing * tick_spacing + tick_spacing)
                    tick_vals = list(range(tick_start, tick_end + 1, tick_spacing))

            fig.update_layout(
                height=500,
                margin=dict(t=40, b=40, l=20, r=20),
                xaxis_title="Time",
                yaxis_title="Normalized Price" if use_normalized else "Price",
                yaxis=dict(
                    gridcolor='rgba(200,200,200,0.5)',
                    gridwidth=1,
                    griddash='dot',
                    tickvals=tick_vals,
                    showgrid=True
                ),
                xaxis=dict(tickangle=-45),
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart data to display.")

        # Tiny debug to see which tickers/timeframes fail
        if empties:
            with st.expander("Chart debug"):
                st.write(f"No data for **{period_label}**:", ", ".join(empties[:20]) + ("…" if len(empties) > 20 else ""))
