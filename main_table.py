# main_table.py
import time
import re
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from Stocks import peter_lynch, personal_model

# Quiet yfinance logs
yf.utils.get_yf_logger().setLevel(logging.ERROR)

# ---------- Fundamentals from cache ----------
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

# ---------- Ticker sanitation ----------
CURRENCY_CODES = {"USD","EUR","GBP","CHF","AUD","NZD","CAD","HKD","JPY","DKK","NOK","SEK","ILS"}

def _looks_like_symbol(t: str) -> bool:
    if not isinstance(t, str) or not t:
        return False
    if t in CURRENCY_CODES:
        return False
    if " " in t:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9.\-\^]+", t))

# ---------- % change helper (calendar offsets from last bar) ----------
def _pct_from_calendar_offset(close: pd.Series, *, months: int = 0, years: int = 0) -> float:
    if close is None or len(close) < 2:
        return np.nan
    close = close.dropna()
    if close.empty:
        return np.nan
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index, errors="coerce")
        close = close.dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if close.empty:
        return np.nan

    end_ts = close.index[-1]
    target = end_ts - pd.DateOffset(years=years, months=months)

    pos = close.index.searchsorted(target, side="left")
    if pos >= len(close):
        pos = close.index.searchsorted(target, side="right") - 1
    if pos < 0:
        return np.nan

    start_px = close.iloc[pos:pos+1].to_numpy()
    end_px   = close.iloc[-1:].to_numpy()
    if start_px.size == 0 or end_px.size == 0 or start_px[0] == 0:
        return np.nan
    return (end_px[0] / start_px[0] - 1.0) * 100.0

# ---------- Close extraction (handles both MultiIndex layouts) ----------
def _get_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    if df is None or df.empty:
        return None

    # Single-index (single ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        s = df.get("Close")
        if s is None or s.empty:
            s = df.get("Adj Close")
        return s.dropna() if s is not None else None

    # MultiIndex: (ticker, field) OR (field, ticker)
    cols = df.columns
    if (ticker, "Close") in cols:
        return df[(ticker, "Close")].dropna()
    if ("Close", ticker) in cols:
        return df[("Close", ticker)].dropna()
    if (ticker, "Adj Close") in cols:
        return df[(ticker, "Adj Close")].dropna()
    if ("Adj Close", ticker) in cols:
        return df[("Adj Close", ticker)].dropna()
    return None

# ---------- Safe wrappers around yfinance ----------
def _download_batch(tickers, period="1y", interval="1d"):
    """Safe batch download. No raise_errors/timeout to support older yfinance."""
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,      # unadjusted to match Yahoo site % moves
            group_by="ticker",      # standard multi-index layout
            threads=True,
            progress=False,
        )
    except Exception:
        return None

def _download_single(ticker, period="1y", interval="1d"):
    """Fallback: single ticker via Ticker.history (more forgiving)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        return df if df is not None and not df.empty else None
    except Exception:
        return None

# ---------- Batch price changes with fallback ----------
def price_changes_batch(tickers, batch_size=60, sleep_between=0.2):
    tickers = [t for t in tickers if _looks_like_symbol(t)]
    out_rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]

        # 1) Batched call
        data = _download_batch(batch)

        found = set()
        if data is not None and not data.empty:
            for t in batch:
                try:
                    close = _get_close_series(data, t)
                    if close is None or close.empty:
                        continue
                    r_1m = _pct_from_calendar_offset(close, months=1)
                    r_1y = _pct_from_calendar_offset(close, years=1)
                    out_rows.append({
                        "Ticker": t,
                        "1M %": None if np.isnan(r_1m) else round(r_1m, 2),
                        "1Y %": None if np.isnan(r_1y) else round(r_1y, 2),
                    })
                    found.add(t)
                except Exception:
                    continue

        # 2) Fallback only for batched misses
        for t in (t for t in batch if t not in found):
            df = _download_single(t)
            if df is None or df.empty:
                time.sleep(0.03)
                continue
            close = _get_close_series(df, t)  # works for single-index too
            if close is None or close.empty:
                time.sleep(0.03)
                continue
            r_1m = _pct_from_calendar_offset(close, months=1)
            r_1y = _pct_from_calendar_offset(close, years=1)
            out_rows.append({
                "Ticker": t,
                "1M %": None if np.isnan(r_1m) else round(r_1m, 2),
                "1Y %": None if np.isnan(r_1y) else round(r_1y, 2),
            })
            time.sleep(0.03)

        time.sleep(sleep_between)

    if not out_rows:
        return pd.DataFrame(columns=["Ticker", "1M %", "1Y %"])
    return pd.DataFrame(out_rows, columns=["Ticker", "1M %", "1Y %"])

# ---------- Scores ----------
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

# ---------- Public builder ----------
def build_quick_table(cache):
    fundamentals = fundamentals_from_cache(cache)
    tickers = sorted(set(fundamentals["Ticker"].dropna()) & set(cache.keys()))
    rets = price_changes_batch(tickers)
    if rets is None or rets.empty:
        rets = pd.DataFrame(columns=["Ticker", "1M %", "1Y %"])
    scores = scores_from_cache(cache)

    table = (
        fundamentals
        .merge(rets, on="Ticker", how="left")
        .merge(scores, on="Ticker", how="left")
    )

    cols = ["Ticker","Name","Sector","P/E","P/B","PEG","D/E","ROIC",
            "Dividend Yield %","1M %","1Y %","Peter Lynch","Personal Model"]
    table = table.reindex(columns=cols)

    num_cols = table.select_dtypes(include=[np.number]).columns
    table[num_cols] = table[num_cols].round(2)
    return table
