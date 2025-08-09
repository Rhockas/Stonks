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

# ========= CHART HELPERS =========
def _clean_series(s: pd.Series) -> pd.Series | None:
    if s is None or s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    # convert to UTC and drop timezone so rangebreaks work predictably
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

def _try_dl(ticker: str, period: str, interval: str):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False, threads=True)
        if df is not None and not df.empty and "Close" in df:
            return df["Close"].dropna()
    except Exception:
        pass
    return None

def fetch_series_for_chart(ticker: str, period_label: str) -> pd.Series | None:
    """
    1D  : 2d/5m  -> slice last 24h
    1W  : 21d/30m -> slice last 7d
    1M  : 60d/60m -> slice last 30d (hourly)
    6M  : 6mo/1d
    1Y  : 1y/1d
    3Y  : 3y/1wk
    """
    try:
        if period_label == "1 Day":
            s = _try_dl(ticker, "2d", "5m")
            return _slice_last(s, "24h")

        if period_label == "1 Week":
            s = _try_dl(ticker, "21d", "30m")
            return _slice_last(s, "7d")

        if period_label == "1 Month":
            # try intraday first
            s = _try_dl(ticker, "60d", "60m") or _try_dl(ticker, "30d", "60m") or _try_dl(ticker, "60d", "90m")
            if s is None or s.empty:
                s = _try_dl(ticker, "1mo", "1d")
            if s is None:
                return None
            s = _clean_series(s)
            if s is None:
                return None
            end = s.index[-1]
            # exact last calendar month from latest date
            start = end - pd.DateOffset(months=1)
            return _clean_series(s[(s.index >= start) & (s.index <= end)])

        if period_label == "6 Months":
            return _clean_series(_try_dl(ticker, "6mo", "1d"))

        if period_label == "1 Year":
            return _clean_series(_try_dl(ticker, "1y", "1d"))

        if period_label == "3 Years":
            return _clean_series(_try_dl(ticker, "3y", "1wk"))
    except Exception:
        return None
    return None

# Simple fallback (like your original)
SIMPLE_MAP = {
    "1 Day":   ("1d",  "5m"),
    "1 Week":  ("5d",  "30m"),
    "1 Month": ("30d", "60m"),
    "6 Months":("6mo", "1d"),
    "1 Year":  ("1y",  "1d"),
    "3 Years": ("3y",  "1wk"),
}
def simple_fetch_series(ticker: str, period_label: str) -> pd.Series | None:
    period, interval = SIMPLE_MAP[period_label]
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is not None and not df.empty and "Close" in df:
            s = _clean_series(df["Close"].dropna())
            # 🔒 ensure exactly last calendar month, even in simple mode
            if period_label == "1 Month" and s is not None and not s.empty:
                end = s.index.max()
                start = end - pd.DateOffset(months=1)
                s = _clean_series(s[(s.index >= start) & (s.index <= end)])
            return s
    except Exception:
        pass
    return None

def intraday_rangebreaks():
    """
    Compress non-trading time for US stocks using UTC hours.
    US regular session ≈ 13:30–20:00 UTC (9:30–16:00 ET).
    Skip 20:00→13:30 and weekends.
    """
    return [
        dict(bounds=["sat", "mon"]),            # skip weekends
        dict(pattern="hour", bounds=[20, 13.5]) # skip 20:00–13:30 UTC
    ]

def is_intraday(label: str) -> bool:
    return label in {"1 Day", "1 Week", "1 Month"}

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
        period_label = st.selectbox(
            "Select time period:",
            ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "3 Years"],
            index=2
        )
        use_normalized = st.checkbox("Normalize (start = 100)", value=True)

        # fetch data with robust -> fallback
        price_data, empties = {}, []
        for t in tickers:
            s = fetch_series_for_chart(t, period_label)
            if s is None or s.empty:
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
                s = price_data[t].astype(float)
                y = (s / s.iloc[0] * 100.0) if use_normalized else s
                pct = (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
                name = f"{t} ({pct:+.2f}%)"

                ymin, ymax = float(y.min()), float(y.max())
                y_min = ymin if y_min is None else min(y_min, ymin)
                y_max = ymax if y_max is None else max(y_max, ymax)

                fig.add_trace(go.Scatter(
                    x=y.index, y=y.values, mode="lines", name=name,
                    hovertemplate=f"{t}<br>Date: %{{x|%Y-%m-%d %H:%M}}<br>"
                                  f"{'Norm ' if use_normalized else ''}Price: %{{y:.2f}}"
                                  "<extra></extra>"
                ))

            layout_kwargs = dict(
                height=500,
                template="plotly_white",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Normalized" if use_normalized else "Price",
                margin=dict(t=30, b=40, l=20, r=20),
                xaxis=dict(tickangle=-45)
            )
            # Only compress off-hours on intraday charts
            if is_intraday(period_label):
                layout_kwargs["xaxis"]["rangebreaks"] = intraday_rangebreaks()
            if selected and period_label == "1 Month":
                end = max(price_data[t].index.max() for t in selected)
                start = end - pd.DateOffset(months=1)
                fig.update_xaxes(range=[start, end])
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart data to display.")

        if empties:
            st.caption(f"No data for timeframe **{period_label}**: {', '.join(empties[:12])}{'…' if len(empties)>12 else ''}")
