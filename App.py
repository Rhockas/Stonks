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
from Stocks import stock_df, method_df  # Updated functions
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Stock Analysis App")

# User input
tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]


st.subheader("Stock Price Chart")

# Time period dropdown
period_map = {
    "1 Day": "1d",
    "1 Week": "5d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}
selected_period_label = st.selectbox("Select time period:", list(period_map.keys()))
selected_period = period_map[selected_period_label]

# Load historical data for each ticker
price_data = {}
for ticker in tickers:
    try:
        df = yf.Ticker(ticker).history(period=selected_period)
        if not df.empty:
            price_data[ticker] = df["Close"]
    except Exception as e:
        st.warning(f"Couldn't load data for {ticker}: {e}")

# Plot all on one chart using normalized prices
if price_data:
    st.markdown("Prices normalized to 100 for comparison.")
    fig, ax = plt.subplots()
    for ticker, series in price_data.items():
        normalized = series / series.iloc[0] * 100
        ax.plot(normalized, label=ticker)
    ax.legend()
    ax.set_ylabel("Normalized Price")
    st.pyplot(fig)
else:
    st.info("No valid price data available to chart.")

def color_final_score(val):
    if isinstance(val, str) and val.endswith('%'):
        try:
            score = float(val.replace('%', ''))
            if score < 20:
                return 'background-color: #8B0000; color: white'
            elif score < 40:
                return 'background-color: #FF6347; color: white'
            elif score < 60:
                return 'background-color: #FFD700; color: black'
            elif score < 80:
                return 'background-color: #9ACD32; color: black'
            else:
                return 'background-color: #228B22; color: white'
        except:
            return ''
    return ''


if st.button("Analyze"):
    if tickers:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stock Financial Metrics")
            df_metrics = stock_df(tickers)
            st.dataframe(df_metrics, use_container_width=True)  # Native sorting enabled

        with col2:
            st.subheader("Valuation Models")
            df_methods = method_df(tickers)
            st.dataframe(df_methods, use_container_width=True)  # Use raw DataFrame for sortability
    else:
        st.warning("Please enter at least one ticker.")

