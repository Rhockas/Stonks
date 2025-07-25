# import streamlit as st
# import pandas as pd
# from Stocks import stock_df  # assuming stock_df(tickers: list) already defined

# st.set_page_config(layout="wide")
# st.title("Stock Analysis App")

# # Input tickers
# tickers_input = st.text_input("Enter tickers (comma-separated):", "GOOGL, ASML, TSLA")
# tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]


# def color_final_score(val):
#     if isinstance(val, str) and val.endswith('%'):
#         try:
#             score = float(val.replace('%', ''))
#             if score < 20:
#                 return 'background-color: #8B0000; color: white'
#             elif score < 40:
#                 return 'background-color: #FF6347; color: white'
#             elif score < 60:
#                 return 'background-color: #FFD700; color: black'
#             elif score < 80:
#                 return 'background-color: #9ACD32; color: black'
#             else:
#                 return 'background-color: #228B22; color: white'
#         except:
#             return ''
#     return ''


# if st.button("Analyze"):
#     if tickers:
#         df = stock_df(tickers)

#         styled_df = df.style.format(precision=2)

#         if "Final Score" in df.columns:
#             styled_df = styled_df.map(color_final_score, subset=["Final Score"])

#         st.dataframe(styled_df, use_container_width=True)
#     else:
#         st.warning("Please enter at least one ticker.")



import streamlit as st
import pandas as pd
from Stocks import stock_df, method_df
import yfinance as yf
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title("Stock Analysis App")

# --- Inputs ---
tickers_input = st.text_input("Enter tickers (comma-separated):", "GOOGL, ASML, AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
analyze = st.button("🔍 Analyze")

# --- Stock Price Chart ---
st.subheader("Stock Price Chart")

# --- Period selection ---
period_map = {
    "1 Day": "1d",
    "1 Week": "5d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}
selected_period_label = st.selectbox("Select time period:", list(period_map.keys()))
selected_period = period_map[selected_period_label]

# --- Toggle for raw vs normalized ---
use_normalized = st.checkbox("Normalize prices (start from 100)", value=True)

# --- Download historical data for each ticker ---
price_data = {}
for ticker in tickers:
    try:
        df = yf.Ticker(ticker).history(period=selected_period)
        if not df.empty:
            price_data[ticker] = df["Close"]
    except Exception as e:
        st.warning(f"Couldn't load data for {ticker}: {e}")

# --- Multiselect filter ---
available_tickers = list(price_data.keys())
selected_tickers = st.multiselect(
    "Select tickers to show in chart:",
    options=available_tickers,
    default=available_tickers
)

# --- Plot chart ---
if selected_tickers:
    fig = go.Figure()
    y_min, y_max = float('inf'), float('-inf')

    for ticker in selected_tickers:
        series = price_data[ticker]
        y_series = series / series.iloc[0] * 100 if use_normalized else series
        y_min = min(y_min, y_series.min())
        y_max = max(y_max, y_series.max())

        fig.add_trace(go.Scatter(
            x=y_series.index,
            y=y_series.values,
            mode='lines',
            name=ticker,
            hovertemplate = (
                f"{ticker}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                + ("Norm " if use_normalized else "")
                + "Price: %{y:.2f}<extra></extra>"
                            )
        ))

    # --- Dynamic gridlines ---
    y_range = y_max - y_min
    tick_spacing = max(round(y_range / 10), 1)
    tick_start = int(y_min // tick_spacing * tick_spacing)
    tick_end = int(y_max // tick_spacing * tick_spacing + tick_spacing)
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
        xaxis=dict(tickangle=-90),
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No tickers selected for chart display.")

# --- Analysis Tables ---
if analyze and tickers:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stock Financial Metrics")
        df_metrics = stock_df(tickers)
        st.dataframe(df_metrics, use_container_width=True)

    with col2:
        st.subheader("Valuation Models")
        df_methods = method_df(tickers)
        st.dataframe(df_methods, use_container_width=True)

elif analyze:
    st.warning("Please enter at least one ticker.")
