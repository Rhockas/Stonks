# App.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

from Stocks import stock_df, stock_data, peter_lynch, personal_model

st.set_page_config(layout="wide", page_title="Stock Quick View")

# ========= Helpers =========
@st.cache_data(ttl=3600)
def load_quick_view(path="quick_view.csv"):
    df = pd.read_csv(path)
    # nicer display: fill NaN with empty, keep 2dp for numbers
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)
    return df.fillna("")

def compute_scores(tickers):
    """Return DataFrame with Peter Lynch & Personal Model scores for given tickers."""
    rows = []
    for t in tickers:
        try:
            d = stock_data(t)         # one call per ticker (manual lookup only)
            rows.append({
                "Ticker": d.get("ticker", t),
                "Peter Lynch": peter_lynch(d),
                "Personal Model": personal_model(d),
            })
        except Exception:
            # Skip bad tickers quietly
            continue
    return pd.DataFrame(rows)

def intraday_15m_series(ticker):
    """15m bars for today; fallback to last trading day from 5d window."""
    df = yf.download(tickers=ticker, period="1d", interval="15m", auto_adjust=True, progress=False)
    if df is not None and not df.empty:
        return df["Close"]
    df5 = yf.download(tickers=ticker, period="5d", interval="15m", auto_adjust=True, progress=False)
    if df5 is None or df5.empty:
        return None
    last_day = df5.index.normalize()[-1]
    return df5.loc[df5.index.normalize() == last_day, "Close"]

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

colA, colB = st.columns(2)
go_btn = colA.button("🔍 Get Details")
chart_btn = colB.button("📈 Show Intraday (15m)")

# --- Details (stock_df + Peter Lynch + Personal Model) ---
if go_btn:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        try:
            df_metrics = stock_df(tickers).copy()  # your existing function
            # Ensure Ticker is present and upper-case for merging
            if "Ticker" not in df_metrics.columns:
                # your stock_df returns Title-cased keys; make sure it has Ticker
                df_metrics.rename(columns={"Ticker": "Ticker"}, inplace=True)
            df_metrics["Ticker"] = df_metrics["Ticker"].astype(str).str.upper()

            df_scores = compute_scores(tickers)
            if not df_scores.empty:
                out = df_metrics.merge(df_scores, on="Ticker", how="left")
            else:
                out = df_metrics

            # round numeric cols to 2dp for display
            num_cols = out.select_dtypes(include=[np.number]).columns
            out[num_cols] = out[num_cols].round(2)

            st.write("**Financial Metrics + Scores**")
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading details: {e}")

# --- Intraday Chart (15m bars, last trading day) ---
if chart_btn:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        st.subheader("Intraday (15m) – last trading day")
        normalize = st.checkbox("Normalize (start = 100)", value=True, key="norm_intraday")
        fig = go.Figure()
        shown = 0

        for t in tickers:
            try:
                s = intraday_15m_series(t)
                if s is None or s.empty:
                    st.info(f"{t}: no intraday data available.")
                    continue
                y = s / s.iloc[0] * 100 if normalize else s
                fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name=t))
                shown += 1
            except Exception as e:
                st.info(f"{t}: failed to load intraday data ({e})")

        if shown:
            fig.update_layout(
                height=450,
                template="plotly_white",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Normalized" if normalize else "Price",
                margin=dict(t=30, b=40, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No intraday data to display.")
