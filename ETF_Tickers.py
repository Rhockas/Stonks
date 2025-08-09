# %%
import pandas as pd
import numpy as np

path = "iShares-Core-MSCI-World-UCITS-ETF_fund.csv"

# %%
def ETF_Ticker_List(path, top_n=None):
    df = pd.read_csv(path, dtype=str)  # keep numeric tickers intact

    # sort by weight (if present)
    if "Weight (%)" in df.columns:
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df = df.dropna(subset=["Weight (%)"]).sort_values("Weight (%)", ascending=False)

    # suffix mapping (your existing logic)
    ticker_dict = {
        "AUD": ".AX","JPY": ".T","CHF": ".SW","GBP": ".L","CAD": ".TO",
        "DKK": ".CO","HKD": ".HK","SGD": ".SI","SEK": ".ST","ILS": ".TA",
        "NOK": ".OL","NZD": ".NZ"
    }
    for cur, suf in ticker_dict.items():
        mask = df["Market Currency"] == cur
        df.loc[mask, "Issuer Ticker"] = (
            df.loc[mask, "Issuer Ticker"].astype(str).str.replace(" ", "-", regex=False) + suf
        )

    tickers = (
        df["Issuer Ticker"].dropna().astype(str).str.strip()
        .replace("", pd.NA).dropna()
        .drop_duplicates()
        .tolist()
    )
    return tickers[:top_n] if top_n else tickers


# %%
# def ETF_Ticker_List():
#     return df["Issuer Ticker"].dropna().tolist()

# ETF_Ticker_List()


