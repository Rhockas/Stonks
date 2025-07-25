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

st.set_page_config(layout="wide")
st.title("Stock Analysis App")

# User input
tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]


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
            st.dataframe(df_metrics, use_container_width=True)

        with col2:
            st.subheader("Valuation Models")
            df_methods = method_df(tickers)
            styled_methods = df_methods.style.format(precision=2)

            if "Final Score" in df_methods.columns:
                styled_methods = styled_methods.map(color_final_score, subset=["Final Score"])

            st.dataframe(styled_methods, use_container_width=True)
    else:
        st.warning("Please enter at least one ticker.")
