import streamlit as st
import pandas as pd
from Stocks import stock_df  # assuming stock_df(tickers: list) already defined

st.title("Stock Analysis App")

# Input tickers
tickers_input = st.text_input("Enter tickers (comma-separated):", "GOOGL, ASML, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Analyze"):
    if tickers:
        df = stock_df(tickers)
        st.dataframe(df.style.format(precision=2))
    else:
        st.warning("Please enter at least one ticker.")
