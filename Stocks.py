# %%
import pandas as pd
import numpy as np
import yfinance as yf

def _num(x):
    # helper to coerce to float safely
    try:
        from math import isfinite
        v = float(x)
        return v if isfinite(v) else None
    except:
        return None

def stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fin = stock.financials
    q_fin = stock.quarterly_financials
    q_bs = stock.quarterly_balance_sheet

    # --- Revenue growth (same structure as your original) ---
    try:
        rev_annual = fin.loc["Total Revenue"].sort_index(ascending=True)
        rev_annual = pd.to_numeric(rev_annual, errors="coerce").dropna()
        if len(rev_annual) >= 4:
            latest_rev = rev_annual.iloc[-1]
            past_rev = rev_annual.iloc[-4]
            if past_rev and past_rev != 0:
                days_elapsed = (pd.Timestamp.today() - pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)).days
                n_years = 3 + days_elapsed / 365.25
                cagr = ((latest_rev / past_rev) ** (1 / n_years) - 1) * 100
                growth_3to4y = ((latest_rev / past_rev) - 1) * 100
            else:
                cagr = growth_3to4y = 0
        else:
            cagr = growth_3to4y = 0
    except Exception:
        cagr = growth_3to4y = 0

    # --- EBIT TTM ---
    try:
        ebit_series = q_fin.loc["EBIT"].iloc[:4]
        ebit_ttm = pd.to_numeric(ebit_series, errors="coerce").fillna(0).sum()
        if ebit_ttm == 0:
            ebit_ttm = None
    except Exception:
        ebit_ttm = None

    # --- Invested Capital ---
    try:
        invested_capital = _num(q_bs.loc["Invested Capital"].iloc[0])
    except Exception:
        invested_capital = None

    # --- Ratios / growth ---
    pe  = _num(info.get("trailingPE"))
    peg = _num(info.get("trailingPegRatio"))
    growth = (pe / peg) if (pe is not None and peg is not None and peg > 0) else 0

    try:
        roic = (ebit_ttm / invested_capital) if (ebit_ttm is not None and invested_capital and invested_capital != 0) else None
    except Exception:
        roic = None

    return {
        "short_name": info.get("shortName"),
        "ticker": ticker.upper(),
        "price": _num(info.get("currentPrice")),
        "pe_ratio": pe,
        "peg_ratio": peg,
        "eps_ttm": _num(info.get("trailingEps")),
        "Total Revenue": _num(info.get("totalRevenue")),
        "growth": growth,
        "3-4 Year Sales Growth": growth_3to4y,
        "CAGR": cagr,
        "EBIT": ebit_ttm,
        "enterprise_value": _num(info.get("enterpriseValue")),
        "invested_capital": invested_capital,
        "ROIC": roic,
        "dividendYield": _num(info.get("dividendYield")),
        "dividend_yield%": _num(info.get("trailingAnnualDividendYield")),
        "earnings_growth": _num(info.get("earningsQuarterlyGrowth")),
        "roe": _num(info.get("returnOnEquity")),
        "roa": _num(info.get("returnOnAssets")),
        "debt_to_equity": (_num(info.get("debtToEquity")) or 0) * 0.01,
        "pb_ratio": _num(info.get("priceToBook")),
        "free_cash_flow": _num(info.get("freeCashflow")),
        "market_cap": _num(info.get("marketCap")),
        "rev_growth": _num(info.get("revenueGrowth")),
        "sector": info.get("sector"),
    }


def safe_compare(val, threshold, comp='>='):
    if val is None:
        return False
    try:
        if comp == '>=':
            return val >= threshold
        elif comp == '<=':
            return val <= threshold
        elif comp == '>':
            return val > threshold
        elif comp == '<':
            return val < threshold
        elif comp == '==':
            return val == threshold
    except:
        return False
    return False

def peter_lynch(data):
    growth = data.get("growth")
    dividend_yield = data.get("dividend_yield%") or 0
    pe_ratio = data.get("pe_ratio")
    score = ((growth * 2 + dividend_yield) / pe_ratio) if growth and pe_ratio else 0
    return round(score, 2)

def warren_buffett(data):
    score = 0
    if safe_compare(data.get("roe"), 0.12): score += 30
    if safe_compare(data.get("debt_to_equity"), 1.0, '<='): score += 20
    if safe_compare(data.get("pe_ratio"), 0, '>') and safe_compare(data.get("pe_ratio"), 30, '<'): score += 30
    if safe_compare(data.get("earnings_growth"), 0, '>'): score += 20
    return score

def philip_fisher(data):
    score = 0
    if safe_compare(data.get("earnings_growth"), 0.1): score += 35
    if safe_compare(data.get("rev_growth"), 0.1, '>'): score += 35
    if safe_compare(data.get("roe"), 0.15, '>'): score += 30
    return score

def magic_formula(data):
    score = 0
    try:
        earn_yield = data['EBIT'] / data['enterprise_value']
    except:
        earn_yield = None
    if safe_compare(earn_yield, 0.07, '>'): score += 50
    if safe_compare(data.get("ROIC"), 0.1, '>'): score += 50
    return score

def seth_klarman(data):
    score = 0
    if safe_compare(data.get("pb_ratio"), 2, '<='): score += 34
    if safe_compare(data.get("free_cash_flow"), 0, '>'): score += 33
    if safe_compare(data.get("debt_to_equity"), 0.5, '<='): score += 33
    return score

def benjamin_graham(data):
    score = 0
    if safe_compare(data.get("pe_ratio"), 20, '<='): score += 20
    if safe_compare(data.get("pb_ratio"), 3, '<='): score += 20
    if safe_compare(data.get("debt_to_equity"), 1.0, '<='): score += 20
    if safe_compare(data.get("dividend_yield%"), 0, '>'): score += 20
    if safe_compare(data.get("earnings_growth"), 0, '>'): score += 20
    return score

def personal_model(data):
    score = 0
    if safe_compare(data.get("roe"), 0.14): score += 12.5
    if safe_compare(data.get("ROIC"), 0.10): score += 12.5
    if safe_compare(data.get("earnings_growth"), 0.12): score += 12.5
    if safe_compare(data.get("growth"), 0.10): score += 12.5
    if safe_compare(data.get("pe_ratio"), 40, '<='): score += 12.5
    if safe_compare(data.get("peg_ratio"), 2.5, '<'): score += 12.5
    if safe_compare(data.get("debt_to_equity"), 0.5, '<='): score += 12.5
    if safe_compare(data.get("free_cash_flow"), 0, '>'): score += 12.5
    return score

def stock_df(tickers):
    rows = []
    for ticker in tickers:
        data = stock_data(ticker)
        row = {k.replace('_', ' ').title(): v for k, v in data.items()}
        rows.append(row)
    return pd.DataFrame(rows).round(2)

def method_df(tickers):
    rows = []
    for ticker in tickers:
        data = stock_data(ticker)
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

        row = {
            "Ticker": data["ticker"],
            'Name': data["short_name"],
            "Peter Lynch": pl_score,
            "Warren Buffett": scores[0],
            "Philip Fisher": scores[1],
            "Magic Formula": scores[2],
            "Benjamin Graham": scores[3],
            "Seth Klarman": scores[4],
            "Personal Model": scores[5],
            "Final Score": f"{final_score}%"
        }
        rows.append(row)
    return pd.DataFrame(rows).round(2)


# %%
stock_df([
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"
]
)

# %%
method_df(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"])



# %%
