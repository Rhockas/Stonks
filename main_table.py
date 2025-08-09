# main_table.py
import pandas as pd
import numpy as np
import yfinance as yf
from Stocks import peter_lynch, personal_model

# --- Fundamentals from cache ---
def fundamentals_from_cache(cache):
    rows = []
    for t, d in cache.items():
        if not d:
            continue
        rows.append({
            "Ticker": d.get("ticker", t),
            "Name": d.get("short_name"),
            "Sector": d.get("sector"),
            "P/E": d.get("pe_ratio"),
            "P/B": d.get("pb_ratio"),
            "PEG": d.get("peg_ratio"),
            "D/E": d.get("debt_to_equity"),
            "ROIC": d.get("ROIC"),
            "Dividend Yield %": d.get("dividend_yield%"),
        })
    return pd.DataFrame(rows)

# --- Batch price changes ---
def price_changes(tickers):
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "1M %", "1Y %"])
    data = yf.download(
        tickers=tickers, period="1y", interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=False
    )
    changes = []
    for t in tickers:
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
            changes.append({
                "Ticker": t,
                "1M %": round(pct_back(21), 2),
                "1Y %": round(pct_back(252), 2),
            })
        except Exception:
            continue
    return pd.DataFrame(changes)

# --- Scores from cache ---
def scores_from_cache(cache):
    rows = []
    for t, d in cache.items():
        if not d:
            continue
        rows.append({
            "Ticker": d.get("ticker", t),
            "Peter Lynch": peter_lynch(d),
            "Personal Model": personal_model(d),
        })
    return pd.DataFrame(rows)

# --- Main quick-view table ---
def build_quick_table(cache):
    fundamentals = fundamentals_from_cache(cache)
    tickers = fundamentals["Ticker"].dropna().tolist()
    rets = price_changes(tickers)
    scores = scores_from_cache(cache)

    table = (fundamentals
             .merge(rets, on="Ticker", how="left")
             .merge(scores, on="Ticker", how="left"))

    cols = [
        "Ticker", "Name", "Sector", "P/E", "P/B", "PEG", "D/E",
        "ROIC", "Dividend Yield %", "1M %", "1Y %",
        "Peter Lynch", "Personal Model"
    ]
    table = table.reindex(columns=cols)

    # Round all numeric columns to 2 decimals
    num_cols = table.select_dtypes(include=[np.number]).columns
    table[num_cols] = table[num_cols].round(2)

    return table