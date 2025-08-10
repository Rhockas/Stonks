# main_table.py
import time
import re
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from Stocks import peter_lynch, personal_model

# Quiet yfinance logs a bit
yf.utils.get_yf_logger().setLevel(logging.ERROR)

# ---------- Fundamentals from cache ----------
def fundamentals_from_cache(cache: dict) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    return df

# ---------- Ticker sanitation ----------
CURRENCY_CODES = {"USD","EUR","GBP","CHF","AUD","NZD","CAD","HKD","JPY","DKK","NOK","SEK","ILS"}

def _looks_like_symbol(t: str) -> bool:
    if not isinstance(t, str) or not t:
        return False
    if t in CURRENCY_CODES:
        return False
    if " " in t:  # e.g., 'NDA FI'
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9.\-\^]+", t))

# ---------- Helpers for % change ----------
def _pct_from_calendar_offset(close: pd.Series, *, months: int = 0, years: int = 0) -> float:
    """
    % change from first Close on/after (last_date - DateOffset(years, months))
    to the last Close. 'close' should be UNADJUSTED closes (auto_adjust=False).
    """
    if close is None or len(close) == 0:
        return np.nan
    # make sure index is datetime & sorted
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.dropna()
    if close.empty:
        return np.nan
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if close.empty:
        return np.nan

    end_ts = close.index[-1]
    target = end_ts - pd.DateOffset(years=years, months=months)

    # first index >= target; else the last index <= target
    pos = close.index.searchsorted(target, side="left")
    if pos >= len(close):
        pos = close.index.searchsorted(target, side="right") - 1
    if pos < 0:
        return np.nan

    start_val = close.iloc[pos:pos+1].to_numpy()
    end_val   = close.iloc[-1:].to_numpy()
    if start_val.size == 0 or end_val.size == 0 or not np.isfinite(start_val[0]) or not np.isfinite(end_val[0]) or start_val[0] == 0:
        return np.nan

    return (end_val[0] / start_val[0] - 1.0) * 100.0

def _download_batch(tickers, period="1y", interval="1d", timeout=25):
    """Wrapper around yf.download without using raise_errors (compat)."""
    return yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,    # unadjusted Close to match Yahoo web % moves
        group_by="ticker",    # multi-index when many tickers
        threads=True,
        progress=False,
        timeout=timeout,
    )

def _get_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Return Close series for ticker from yf.download output (single or multi)."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in set(df.columns.get_level_values(0)):
            return None
        s = df.get((ticker, "Close"))
        return s.dropna() if s is not None else None
    # single ticker case
    s = df.get("Close")
    return s.dropna() if s is not None else None

# ---------- Batch price changes ----------
def price_changes_batch(tickers, batch_size=75, sleep_between=0.25) -> pd.DataFrame:
    """
    Compute 1M% and 1Y% using UNADJUSTED daily Close, in batches.
    Skips bad/missing symbols quietly.
    Returns DataFrame['Ticker','1M %','1Y %'] with float dtypes.
    """
    tickers = [t for t in tickers if _looks_like_symbol(t)]
    out_rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = _download_batch(batch, period="1y", interval="1d", timeout=25)
        except Exception:
            time.sleep(sleep_between)
            continue

        # Figure out which symbols actually came back
        present = set()
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            present = set(data.columns.get_level_values(0))
        elif isinstance(data, pd.DataFrame) and "Close" in data.columns and len(batch) == 1:
            present = {batch[0]}

        for t in present:
            try:
                close = _get_close_series(data, t)
                if close is None or close.empty:
                    continue

                r_1m = _pct_from_calendar_offset(close, months=1)
                r_1y = _pct_from_calendar_offset(close, years=1)

                out_rows.append({
                    "Ticker": t,
                    "1M %": np.nan if np.isnan(r_1m) else float(round(r_1m, 2)),
                    "1Y %": np.nan if np.isnan(r_1y) else float(round(r_1y, 2)),
                })
            except Exception:
                continue

        time.sleep(sleep_between)

    if not out_rows:
        return pd.DataFrame(columns=["Ticker", "1M %", "1Y %"]).astype({"Ticker": "string", "1M %": "float64", "1Y %": "float64"})
    df = pd.DataFrame(out_rows, columns=["Ticker", "1M %", "1Y %"])
    # Ensure dtypes
    df["Ticker"] = df["Ticker"].astype("string")
    df["1M %"] = pd.to_numeric(df["1M %"], errors="coerce")
    df["1Y %"] = pd.to_numeric(df["1Y %"], errors="coerce")
    return df

# ---------- Scores ----------
def scores_from_cache(cache: dict) -> pd.DataFrame:
    rows = []
    for t, d in cache.items():
        if not d:
            continue
        rows.append({
            "Ticker": d.get("ticker", t),
            "Peter Lynch": peter_lynch(d),
            "Personal Model": personal_model(d),
        })
    df = pd.DataFrame(rows)
    # enforce numeric types for scores
    for c in ("Peter Lynch", "Personal Model"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype("string")
    return df

# ---------- Build table (data only) ----------
NUMERIC_COLS = ["P/E","P/B","PEG","D/E","ROIC","Dividend Yield %","1M %","1Y %","Peter Lynch","Personal Model"]

def _enforce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force known numeric columns to real numeric dtypes; leave strings as strings."""
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype("string")
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype("string")
    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].astype("string")
    return df

def build_quick_table(cache: dict) -> pd.DataFrame:
    fundamentals = fundamentals_from_cache(cache)
    fundamentals = _enforce_numeric(fundamentals)

    tickers = sorted(set(fundamentals["Ticker"].dropna().astype(str)) & set(cache.keys()))
    rets = price_changes_batch(tickers)
    scores = scores_from_cache(cache)

    table = (fundamentals
             .merge(rets, on="Ticker", how="left")
             .merge(scores, on="Ticker", how="left"))

    # Final column order
    cols = ["Ticker","Name","Sector","P/E","P/B","PEG","D/E","ROIC","Dividend Yield %","1M %","1Y %","Peter Lynch","Personal Model"]
    table = table.reindex(columns=cols)

    # Keep numeric as numeric (round without casting to string)
    for c in NUMERIC_COLS:
        if c in table.columns:
            table[c] = table[c].round(2)

    return table
