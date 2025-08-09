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

# ========= CHART HELPERS (only part changed) =========
def _clean_series(s: pd.Series):
    if s is None or s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if len(s) > 1 else None

def fetch_series_for_chart(ticker: str, period_label: str):
    """
    Original behavior for all periods, with a reliable 1‑Day:
    - 1 Day: use 5m bars from last 2 days, slice last 24h
    - Other periods: use Ticker.history(period=...), fallback to yf.download
    """
    try:
        if period_label == "1 Day":
            df = yf.download(ticker, period="2d", interval="5m",
                             auto_adjust=False, progress=False, threads=True)
            if df is None or df.empty or "Close" not in df:
                return None
            s = df["Close"].dropna()
            end = s.index[-1]
            start = end - pd.Timedelta("24h")
            return _clean_series(s[(s.index >= start) & (s.index <= end)])

        period_map = {
            "1 Week": "5d",
            "1 Month": "1mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "3 Years": "3y",
        }
        period = period_map[period_label]

        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if (df is None or df.empty) and period_label in {"1 Month", "6 Months", "1 Year"}:
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=False, progress=False, threads=True)
        if (df is None or df.empty) and period_label == "1 Week":
            df = yf.download(ticker, period="5d", interval="30m",
                             auto_adjust=False, progress=False, threads=True)
        if (df is None or df.empty) and period_label == "3 Years":
            df = yf.download(ticker, period="3y", interval="1wk",
                             auto_adjust=False, progress=False, threads=True)

        if df is None or df.empty or "Close" not in df:
            return None
        return _clean_series(df["Close"])
    except Exception:
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

# ========= Chart (original look & feel, now reliable) =========
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

        # gather data
        price_data = {}
        for t in tickers:
            s = fetch_series_for_chart(t, period_label)
            if s is not None:
                price_data[t] = s

        selected = st.multiselect(
            "Select tickers to show in chart:",
            options=list(price_data.keys()),
            default=list(price_data.keys())
        )

        if selected:
            fig = go.Figure()
            y_min, y_max = float('inf'), float('-inf')

            for t in selected:
                series = price_data[t]
                y_series = series / series.iloc[0] * 100 if use_normalized else series
                y_min = min(y_min, y_series.min())
                y_max = max(y_max, y_series.max())

                fig.add_trace(go.Scatter(
                    x=y_series.index,
                    y=y_series.values,
                    mode='lines',
                    name=t,
                    hovertemplate = (
                        f"{t}<br>"
                        "Date: %{x|%Y-%m-%d %H:%M}<br>"
                        + ("Norm " if use_normalized else "")
                        + "Price: %{y:.2f}<extra></extra>"
                    )
                ))

            y_range = y_max - y_min
            tick_spacing = max(round(y_range / 10), 1) if y_range != float('inf') else 1
            tick_start = int(y_min // tick_spacing * tick_spacing) if y_min != float('inf') else 0
            tick_end = int(y_max // tick_spacing * tick_spacing + tick_spacing) if y_max != float('-inf') else 1
            tick_vals = list(range(tick_start, tick_end + 1, tick_spacing))

            fig.update_layout(
                height=500,
                margin=dict(t=40, b=40, l=20, r=20),
                xaxis_title="Date",
                yaxis_title="Normalized Price" if use_normalized else "Raw Price",
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
            st.info("No tickers selected for chart display.")
