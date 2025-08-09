# %%
from Stocks import stock_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time, os, random

from Stocks import (
    peter_lynch, warren_buffett, philip_fisher, magic_formula,
    benjamin_graham, seth_klarman, personal_model
)

# ---------- table builders (unchanged) ----------
def stock_df_from_cache(data_by_ticker):
    rows = []
    for t, data in data_by_ticker.items():
        if not data:
            continue
        ticker_val = data.get("ticker", t)
        # start with ticker, then rest of the fields
        row = {"Ticker": ticker_val}
        row.update({k.replace('_', ' ').title(): v for k, v in data.items()})
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def method_df_from_cache(data_by_ticker):
    rows = []
    for t, data in data_by_ticker.items():
        if not data:
            continue
        # use key as fallback
        ticker = data.get("ticker", t)

        pl_score = peter_lynch(data)
        scores = [
            warren_buffett(data),
            philip_fisher(data),
            magic_formula(data),
            benjamin_graham(data),
            seth_klarman(data),
            personal_model(data)
        ]
        avg_score = np.mean(scores)

        if 0.5 <= pl_score < 1: pl_adj = 20
        elif 1 <= pl_score < 1.5: pl_adj = 50
        elif 1.5 <= pl_score < 2: pl_adj = 75
        elif pl_score >= 2: pl_adj = 100
        else: pl_adj = 0

        final_score = round((avg_score * 6 + pl_adj) / 7, 2)

        rows.append({
            "Ticker": ticker,
            "Name": data.get("short_name"),
            "Peter Lynch": pl_score,
            "Warren Buffett": scores[0],
            "Philip Fisher": scores[1],
            "Magic Formula": scores[2],
            "Benjamin Graham": scores[3],
            "Seth Klarman": scores[4],
            "Personal Model": scores[5],
            "Final Score": f"{final_score}%"
        })
    return pd.DataFrame(rows).round(2)


# ---------- retry wrapper to survive YF rate limits ----------
def _with_retry(fn, *args, **kwargs):
    import time, random
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "404" in msg:  # skip dead tickers
                return None
            if "Too Many Requests" in msg or "Rate limited" in msg:
                sleep_s = 15 + attempt * 10 + random.uniform(0, 3)
            else:
                sleep_s = (2 ** attempt) * 0.3 + random.uniform(0, 0.5)
            if attempt >= 2:
                print(f"Retry {attempt}: {args[0]} — {msg} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)
    return None



# ---------- bulk fetch with controlled concurrency ----------
def fetch_stock_data_bulk(tickers, max_workers=6):
    """
    Fetch per-ticker data with retries and modest concurrency to avoid
    Yahoo rate-limits. Returns dict[ticker] -> dict | None
    """
    results = {}
    if not tickers:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_with_retry, stock_data, t): t for t in tickers
        }
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except Exception as e:
                # Should be rare because _with_retry already swallowed errors
                print(f"Failed: {t} — {e}")
                results[t] = None
    return results


# ---------- parquet I/O with sanitization ----------
def save_cache_parquet(data_by_ticker, path="etf_cache.parquet"):
    df = pd.DataFrame.from_dict(data_by_ticker, orient="index")

    # Avoid duplicate 'ticker' column conflicting with index name
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    # Sanitize bad values so Parquet is happy
    df = df.replace([np.inf, -np.inf, "Infinity", "-Infinity"], np.nan)

    # Coerce numeric-looking object cols where possible
    for col in df.columns:
        if df[col].dtype == "object":
            try:
             df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    df = df.convert_dtypes()
    df.index.name = "ticker"
    df.to_parquet(path)  # requires pyarrow


def load_cache_parquet_if_fresh(path="etf_cache.parquet", max_age_hours=24):
    if not os.path.exists(path):
        return None
    age_h = (time.time() - os.path.getmtime(path)) / 3600
    if age_h > max_age_hours:
        return None
    df = pd.read_parquet(path)  # index is 'ticker'
    return df.to_dict(orient="index")

# %%
