# %% [markdown]
# # General Idea
# 1. User enters a stock ticker.
# 2. System retrieves stock’s key financial data.
# 3. For each investing method (Lynch, Buffett, Fisher, etc.):
#     * Evaluate whether the stock passes or not.
#     * Show calculation details: e.g., P/E, PEG, EPS growth, ROE, moat flags, debt/equity etc.
# 4. Aggregate results & output a final suggestion:
#     * e.g., “GOOGL looks reasonable to buy — passes 4/6 methods”
# 5. Optionally: support evaluating multiple tickers at once, outputting a table or report.
# 
# # Multiple stock evaluation methods checker
# 1. function to get the input stock info
# 2. function to output method evaluations for each method
# 3. function to input stock and get all evaluations + 

# %%
import pandas as pd
import numpy as np
import yfinance as yf

# %%
def stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    data = {
        'short_name': info.get('shortName'),
        'ticker': ticker.upper(),
        'price': info.get('currentPrice'),
        'pe_ratio': (info.get('trailingPE')or 1),
        'peg_ratio': (info.get("trailingPegRatio") or 1),
        'eps_ttm': info.get('trailingEps'),
        'growth': 0,
        'growth_with_divs': 0,
        'EPS growth': 0,
        'dividendYield': info.get("dividendYield"),
        'dividend_yield%': (info.get("trailingAnnualDividendYield") or 0.0),
        'earnings_growth': (info.get('earningsQuarterlyGrowth') or 0.0),
        'roe': (info.get('returnOnEquity') or 0.0),
        'roa': info.get('returnOnAssets'),
        'debt_to_equity': (info.get('debtToEquity') or 0.0) * 0.01,
        'pb_ratio': info.get('priceToBook'),
        'free_cash_flow': (info.get('freeCashflow') or 0.0),
        'market_cap': info.get('marketCap'),
        'rev_growth': info.get("revenueGrowth"),
        'sector': info.get('sector')
    }
    if data["peg_ratio"] and data["peg_ratio"] > 0:
        data["growth"] = data["pe_ratio"] / data["peg_ratio"]
    else:
        data["growth"] = 0
    
    return data


# %%
stock_data("GOOGL")

# %%
def peter_lynch(ticker):
    data = stock_data(ticker)
    score = ((data["growth"]*2) + data["dividend_yield%"]) / data["pe_ratio"]
    eval = ""
    result = []
    result.append(f"{score:.2f}")
    if score < 1:
        eval += "overvalued❌"
    elif score >= 1 and score <2:
        eval += "fair price✅"
    elif score >= 2 and score <3:
        eval += "undervalued✅"
    elif score >3:
        "might be wrong."
    #return result
    return f"{score:.2f}"
    return f"\nPeter Lynch: {score:.2f} | {eval}"
peter_lynch("GOOGL")

# %%
def warren_buffett(ticker):
    data = stock_data(ticker)
    roe = data["roe"]
    de = data["debt_to_equity"]
    pe = data["pe_ratio"]
    earn = data["earnings_growth"]
    verdict = "PASSED✅" if (roe >= 0.12) and (de <= 1.0) and (pe > 0 and pe < 30) and (earn > 0) else "DID NOT PASS❌"
    result = []
    score = 0
    if roe >= 0.12:
        result.append("ROE: Good✅\n")
        score += 30
    else:
        result.append("ROE: Weak❌")

    if de <= 1.0:
        result.append("Debt: Low✅")
        score += 20
    else:
        result.append("Debt: High❌")

    if 0 < pe < 30:
        result.append("PE: Fair✅")
        score += 30
    else:
        result.append("PE: High❌")

    if earn > 0:
        result.append("Growth: Positive✅")
        score +=20
    else:
        result.append("Growth: Negative❌")

    return f"{score}%"
    return f"{verdict} | {' | '.join(result)}"

    return f"\nWarren Buffett: {verdict} | {' | '.join(result)}"

warren_buffett("GOOGL")

# %%
def philip_fisher(ticker):
    data = stock_data(ticker)
    rev_growth = data["rev_growth"]
    earn_growth = data["earnings_growth"]
    roe = data["roe"]
    score = 0

    if (earn_growth >= 0.1) and (rev_growth > 0.1) and (roe > 0.12):
        verdict = "PASSED✅"
    else:
        verdict = "DID NOT PASS❌"

    details = []
    if earn_growth >= 0.1:
        details.append("Earnings Growth: Good✅")
        score += 35
    else:
        details.append("Earnings Growth: Weak❌")

    if rev_growth > 0.1:
        details.append("Sales Growth: Good✅")
        score += 35
    else:
        details.append("Sales Growth: Weak❌")

    if roe > 0.12:
        details.append("ROE: Strong✅")
        score +=30
    else:
        details.append("ROE: Weak❌")

    result = (
        f"\nPhilip Fisher: {verdict} | {' | '.join(details)}"
    )
    return f"{score}%"
    return result

philip_fisher("GOOGL")

# %%
def magic_formula(ticker):
    data = stock_data(ticker)
    earn_yield = data["eps_ttm"] / data["price"]
    roa = data["roa"]
    earn_yield_gap = int((earn_yield / 0.1 - 1) * 100)
    roa_gap = int((roa / 0.2 - 1) * 100)
    score = 0
    if (earn_yield > 0.07) and (roa > 0.1):
        verdict = "PASSED✅"
    else:
        verdict = "DID NOT PASS❌"

    details = []
    if earn_yield > 0.07:
        details.append(f"Earnings Yield: Good✅ ({earn_yield:.3f})")
        score += 50
    else:
        details.append(f"Earnings Yield: Weak❌ ({earn_yield:.3f})")

    if roa > 0.1:
        details.append(f"ROA: Strong✅ ({roa:.3f})")
        score+=50
    else:
        details.append(f"ROA: Weak❌ ({roa:.3f})")

    result = f"\nMagic Formula: {verdict} | " + " | ".join(details)
    return f"{score}%"
    return result
magic_formula("GOOGL")

# %%
def seth_klarman(ticker):
    data = stock_data(ticker)
    pb = data["pb_ratio"]
    free_cash_flow = data["free_cash_flow"]
    de = data["debt_to_equity"]
    score = 0
    if (pb <= 2) and (free_cash_flow > 0) and (de <= 0.5):
        verdict = "PASSED ✅"
    else:
        verdict = "DID NOT PASS ❌"

    details = []
    if pb <= 2:
        details.append(f"P/B: Low✅ ({pb:.2f})")
        score += 34
    else:
        details.append(f"P/B: High❌ ({pb:.2f})")

    if free_cash_flow > 0:
        details.append(f"FCF: Positive✅ ({free_cash_flow:.0f})")
        score += 33
    else:
        details.append(f"FCF: Negative❌ ({free_cash_flow:.0f})")

    if de <= 0.5:
        details.append(f"Debt/Equity: Low✅ ({de:.2f})")
        score += 33
    else:
        details.append(f"Debt/Equity: High❌ ({de:.2f})")

    result = f"\nSeth Klarman: {verdict} | " + " | ".join(details)
    return f"{score}%"
    return result
seth_klarman("GOOGL")


# %%
def benjamin_graham(ticker):
    data = stock_data(ticker)
    pe = data["pe_ratio"]
    pb = data["pb_ratio"]
    de = data["debt_to_equity"]
    div_yield = data["dividend_yield%"]
    earn_growth = data["earnings_growth"]
    score = 0
    if (
        (pe <= 20.0)
        and (pb <= 3.0)
        and (de <= 1.0)
        and (div_yield > 0)
        and (earn_growth > 0)
    ):
        verdict = "PASSED ✅"
    else:
        verdict = "DID NOT PASS ❌"

    details = []

    if pe <= 20.0:
        details.append(f"P/E: Good✅ ({pe:.2f})")
        score += 20
    else:
        details.append(f"P/E: High❌ ({pe:.2f})")

    if pb <= 3.0:
        details.append(f"P/B: Good✅ ({pb:.2f})")
        score += 20
    else:
        details.append(f"P/B: High❌ ({pb:.2f})")

    if de <= 1.0:
        details.append(f"D/E: Good✅ ({de:.2f})")
        score += 20
    else:
        details.append(f"D/E: High❌ ({de:.2f})")

    if div_yield > 0:
        details.append(f"Dividend: Present✅ ({div_yield:.2f}%)")
        score += 20
    else:
        details.append(f"Dividend: None❌ ({div_yield:.2f}%)")

    if earn_growth > 0:
        details.append(f"Earnings Growth: Positive✅ ({earn_growth:.2f})")
        score += 20
    else:
        details.append(f"Earnings Growth: Negative❌ ({earn_growth:.2f})")

    result = f"\nBenjamin Graham: {verdict} | " + " | ".join(details)
    return f"{score}%"
    return result

benjamin_graham("GOOGL")


# %%
def personal_model(ticker):
    data = stock_data(ticker)
    roe = data["roe"]
    roa = data["roa"]
    earnings_growth = data["earnings_growth"]
    revenue_growth = data["growth"]
    pe = data["pe_ratio"]
    peg = data["peg_ratio"]
    de = data["debt_to_equity"]
    fcf = data["free_cash_flow"]
    details = []
    score = 0

    if roe >= 0.10:
        score += 10
        details.append(f"ROE: {roe:.2f} ✅")
    else:
        details.append(f"ROE: {roe:.2f} ❌")
    if roa >= 0.05:
        score += 10
        details.append(f"ROA: {roa:.2f} ✅")
    else:
        details.append(f"ROA: {roa:.2f} ❌")

    if earnings_growth >= 0.12:
        score += 20
        details.append(f"Earnings Growth: {earnings_growth:.2f} ✅")
    else:
        details.append(f"Earnings Growth: {earnings_growth:.2f} ❌")
    if revenue_growth >= 0.10:
        score += 20
        details.append(f"Revenue Growth: {revenue_growth:.2f} ✅")
    else:
        details.append(f"Revenue Growth: {revenue_growth:.2f} ❌")

    if pe <= 40:
        score += 10
        details.append(f"P/E: {pe:.2f} ✅")
    else:
        details.append(f"P/E: {pe:.2f} ❌")
    if peg < 2.5:
        score += 10
        details.append(f"PEG: {peg:.2f} ✅")
    else:
        details.append(f"PEG: {peg:.2f} ❌")

    if de <= 1.5:
        score += 10
        details.append(f"D/E: {de:.2f} ✅")
    else:
        details.append(f"D/E: {de:.2f} ❌")
    if fcf > 0:
        score += 10
        details.append(f"FCF: {fcf:.0f} ✅")
    else:
        details.append(f"FCF: {fcf:.0f} ❌")
    return f"{score}%"
    return f"\nPersonal Model — Total Score: {score}/100\n" + "\n".join(details)

personal_model("NVDA")


# %%
def method_evaluation(ticker):
    data = stock_data(ticker)
    for key, value in data.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")

    print("**Methods**")
    print(peter_lynch(ticker))
    print(warren_buffett(ticker))
    print(philip_fisher(ticker))
    print(magic_formula(ticker))
    print(benjamin_graham(ticker))
    print(seth_klarman(ticker))
    print(personal_model(ticker))


# %%
import yfinance as yf

def list_info_fields(ticker_str):
    ticker = yf.Ticker(ticker_str)
    info = ticker.info

    print(f"Available fields for {ticker_str}:")
    for key in sorted(info.keys()):
        print(f"- {key}")

list_info_fields("GOOGL")


# %%
method_evaluation("TSLA")
# print(method_evaluation("MSFT"))
# print(method_evaluation("NVDA"))
# print(method_evaluation("ASML"))
# print(method_evaluation("TSLA"))
# print(method_evaluation("META"))
# print(method_evaluation("GOOGL"))
# print(method_evaluation("AAPL"))
# print(method_evaluation("NET"))
# print(method_evaluation("CRWD"))

# %%
def stock_df(tickers: list):
    rows = []
    for ticker in tickers:
        data = stock_data(ticker)
        row = {
            "Ticker": ticker,
            "Short Name": data["short_name"],
            "Price": data["price"],
            "PE": data["pe_ratio"],
            "PEG": data["peg_ratio"],
            "EPS TTM": data["eps_ttm"],
            "Earnings Growth": data["earnings_growth"],
            "Revenue Growth": data["rev_growth"],
            "ROE": data["roe"],
            "ROA": data["roa"],
            "Debt/Equity": data["debt_to_equity"],
            "P/B": data["pb_ratio"],
            "FCF": data["free_cash_flow"],
            "Dividend Yield %": data["dividend_yield%"]*100,
            "Sector": data["sector"]
        }

        # Clean method outputs
        def clean(method_output):
            return method_output.strip().replace('\n', ' ')

        row["Peter Lynch"] = clean(peter_lynch(ticker))
        row["Warren Buffett"] = clean(warren_buffett(ticker))
        row["Philip Fisher"] = clean(philip_fisher(ticker))
        row["Magic Formula"] = clean(magic_formula(ticker))
        row["Benjamin Graham"] = clean(benjamin_graham(ticker))
        row["Seth Klarman"] = clean(seth_klarman(ticker))
        row["Personal Model"] = clean(personal_model(ticker))
        # Personal Model
        # pm_lines = personal_model(ticker).splitlines()
        # if len(pm_lines) >= 2:
        #     row["Personal Model"] = ' | '.join(pm_lines[1:]).replace('\n', ' ')
        # else:
        #     row["Personal Model"] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.round(2)
    return df


# %%
pd.options.display.max_columns = None
df = stock_df(["GOOGL", "ASML", "TSLA", "MSFT", "AMZN", "AAPL"])
df.head()


