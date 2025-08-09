# main_table.py
import pandas as pd
import numpy as np
import yfinance as yf
from Stocks import peter_lynch, personal_model
import time
import re

def fundamentals_from_cache(cache):
    rows = []
    for t, d in cache.items():
        if not d: continue
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

CURRENCY_CODES = {"USD","EUR","GBP","CHF","AUD","NZD","CAD","HKD","JPY","DKK","NOK","SEK","ILS"}

def _looks_like_symbol(t: str) -> bool:
    if not isinstance(t, str) or not t:
        return False
    if t in CURRENCY_CODES:
        return False
    if " " in t:  # e.g., 'NDA FI'
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9\.\-\^]+", t))

def _pct_from_calendar_offset(close: pd.Series, *, months: int = 0, years: int = 0) -> float:
    """
    % change from the first close on/after (last_date - DateOffset(years, months)) to the last close.
    Uses unadjusted Close semantics (the Series passed in should be unadjusted).
    """
    if close is None or len(close) == 0:
        return np.nan
    close = close.dropna()
    if close.empty:
        return np.nan

    end_ts = close.index[-1]
    target = end_ts - pd.DateOffset(years=years, months=months)

    # find first index >= target; if none, fall back to last index <= target
    pos = close.index.searchsorted(target, side="left")
    if pos >= len(close):
        # no value on/after target; take the latest prior
        pos = close.index.searchsorted(target, side="right") - 1
    if pos < 0:
        return np.nan

    start_px = float(close.iloc[pos])
    end_px = float(close.iloc[-1])
    if start_px == 0 or not np.isfinite(start_px) or not np.isfinite(end_px):
        return np.nan
    return (end_px / start_px - 1.0) * 100.0


# --- replace price_changes_batch with this version (calendar offsets + unadjusted Close) ---
def price_changes_batch(tickers, batch_size=200, sleep_between=0.6):
    """
    Compute 1M% (calendar 1 month) and 1Y% (calendar 1 year) in batches.
    Uses UNADJUSTED Close to better match Yahoo site numbers.
    Always returns DataFrame['Ticker','1M %','1Y %'].
    """
    tickers = [t for t in tickers if _looks_like_symbol(t)]
    out_rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,   # <-- unadjusted Close
                threads=True,
                progress=False,
            )
        except Exception:
            time.sleep(sleep_between)
            continue

        # single-ticker vs multi-ticker
        single_frame = isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex)

        for t in batch:
            try:
                if single_frame:
                    close = data.get("Close")
                else:
                    if not isinstance(data.columns, pd.MultiIndex):
                        continue
                    # guard for partial batches
                    if t not in set(data.columns.get_level_values(0)):
                        continue
                    close = data[(t, "Close")]

                if close is None:
                    continue
                close = close.dropna()
                if close.empty:
                    continue

                r_1m = _pct_from_calendar_offset(close, months=1)
                r_1y = _pct_from_calendar_offset(close, years=1)

                out_rows.append({
                    "Ticker": t,
                    "1M %": None if np.isnan(r_1m) else round(r_1m, 2),
                    "1Y %": None if np.isnan(r_1y) else round(r_1y, 2),
                })
            except Exception:
                continue

        time.sleep(sleep_between)

    if not out_rows:
        return pd.DataFrame(columns=["Ticker", "1M %", "1Y %"])
    return pd.DataFrame(out_rows, columns=["Ticker", "1M %", "1Y %"])

def scores_from_cache(cache):
    rows = []
    for t, d in cache.items():
        if not d: continue
        rows.append({
            "Ticker": d.get("ticker", t),
            "Peter Lynch": peter_lynch(d),
            "Personal Model": personal_model(d),
        })
    return pd.DataFrame(rows)

def build_quick_table(cache):
    fundamentals = fundamentals_from_cache(cache)
    tickers = sorted(set(fundamentals["Ticker"].dropna()) & set(cache.keys()))
    rets = price_changes_batch(tickers)
    if rets is None or rets.empty:
        rets = pd.DataFrame(columns=["Ticker","1M %","1Y %"])
    scores = scores_from_cache(cache)

    table = (fundamentals
             .merge(rets, on="Ticker", how="left")
             .merge(scores, on="Ticker", how="left"))

    cols = ["Ticker","Name","Sector","P/E","P/B","PEG","D/E","ROIC",
            "Dividend Yield %","1M %","1Y %","Peter Lynch","Personal Model"]
    table = table.reindex(columns=cols)

    # Round everything numeric to 2dp
    num_cols = table.select_dtypes(include=[np.number]).columns
    table[num_cols] = table[num_cols].round(2)
    return table
