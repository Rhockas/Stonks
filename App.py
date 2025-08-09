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
    df = pd.read_csv(path)
    return df  # keep raw; we’ll format later

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

def fill_missing_price_changes(df_quick: pd.DataFrame) -> pd.DataFrame:
    """
    If 1M% / 1Y% are missing, compute them once in a batch and merge.
    """
    out = df_quick.copy()
    # Ensure the columns exist
    for col in ["1M %", "1Y %"]:
        if col not in out.columns:
            out[col] = np.nan

    missing = out.loc[out["1M %"].isna() | out["1Y %"].isna(), "Ticker"].dropna().astype(str).tolist()
    if not missing:
        return out

    try:
        data = yf.download(
            tickers=list(set(missing)),
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        rows = []
        for t in missing:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data[t]["Close"].dropna()
                else:
                    close = data["Close"].dropna()
                if close.empty:
                    continue
                last = close.iloc[-1]
                def pct_back(n):
                    return float((last / close.iloc[-n] - 1) * 100) if len(close) > n else np.nan
                rows.append({"Ticker": t, "1M %": round(pct_back(21), 2), "1Y %": round(pct_back(252), 2)})
            except Exception:
                continue
        if rows:
            patch = pd.DataFrame(rows)
            out = out.merge(patch, on="Ticker", how="left", suffixes=("", "_new"))
            for col in ["1M %", "1Y %"]:
                out[col] = out[col].where(out[col].notna(), out[f"{col}_new"])
                if f"{col}_new" in out.columns:
                    out.drop(columns=[f"{col}_new"], inplace=True)
    except Exception:
        pass
    return out

# ---- Chart helpers ----
def get_series_for_timeframe(ticker, tf_label):
    """
    Fetch exactly the requested window ending at the LATEST timestamp we have.
    1D = last 24 hours (slice from 2d/5m so weekends still return a stable last-day line)
    1W = last 7 days (slice from 1mo/30m)
    Others = direct fetch
    """
    try:
        if tf_label == "1 Day":
            d = yf.download(ticker, period="2d", interval="5m", auto_adjust=False, progress=False)
            if d is None or d.empty:
                return None
            s = d["Close"].dropna()
            end = s.index[-1]
            start = end - pd.Timedelta("24h")
            return s[(s.index >= start) & (s.index <= end)]

        elif tf_label == "1 Week":
            d = yf.download(ticker, period="1mo", interval="30m", auto_adjust=False, progress=False)
            if d is None or d.empty:
                return None
            s = d["Close"].dropna()
            end = s.index[-1]
            start = end - pd.Timedelta("7d")
            return s[(s.index >= start) & (s.index <= end)]

        elif tf_label == "1 Month":
            d = yf.download(ticker, period="1mo", interval="1d", auto_adjust=False, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "6 Months":
            d = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "1 Year":
            d = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

        elif tf_label == "3 Years":
            d = yf.download(ticker, period="3y", interval="1wk", auto_adjust=False, progress=False)
            return None if d is None or d.empty else d["Close"].dropna()

    except Exception:
        return None
    return None

# ========= Header =========
st.title("Stock Quick View")

# ========= Quick View Table =========
st.subheader("Quick View (sortable)")
quick_df = load_quick_view()
quick_df = fill_missing_price_changes(quick_df)

# Display formatting: 2dp for numerics, empty strings for NaN
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

            # make index tz-naive for Plotly if tz exists (cleaner hover)
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

        if empty_list:
            st.caption(f"No data returned for: {', '.join(empty_list[:10])}{'…' if len(empty_list)>10 else ''}")
