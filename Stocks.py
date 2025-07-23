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
    
    fin = stock.financials
    q_fin = stock.quarterly_financials
    q_bs = stock.quarterly_balance_sheet


    rev_annual = fin.loc["Total Revenue"]
    rev_annual = rev_annual.sort_index(ascending=True)

    latest_rev = rev_annual.iloc[-1]
    y3to4_years_ago_rev = rev_annual.iloc[-4]

    growth_3to4y = ((latest_rev / y3to4_years_ago_rev) - 1) * 100

    today = pd.Timestamp.today()
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    days_elapsed = (today - start_of_year).days
    n_years = 3 + days_elapsed / 365.25

    ebit_ttm = q_fin.loc["EBIT"].iloc[:4].sum()
    invested_capital = q_bs.loc["Invested Capital"].iloc[0]

    data = {
        'short_name': info.get('shortName'),
        'ticker': ticker.upper(),
        'price': info.get('currentPrice'),
        'pe_ratio': info.get('trailingPE'),
        'peg_ratio': info.get("trailingPegRatio"),
        'eps_ttm': info.get('trailingEps'),
        'Total Revenue': info.get('totalRevenue'),
        'growth': 0,
        'growth_with_divs': 0,
        'EPS growth': 0,
        '3-4 Year Sales Growth': float(growth_3to4y),
        'CAGR': float(((latest_rev / y3to4_years_ago_rev) ** (1/n_years) - 1) * 100),
        'EBIT': float(ebit_ttm),
        'Enterprise Value': info.get("enterpriseValue"),
        'Invested Capital': float(invested_capital),
        "ROIC": (float(ebit_ttm/invested_capital)) if ebit_ttm is not None and invested_capital is not None else "N/A",
        'dividendYield': info.get("dividendYield"),
        'dividend_yield%': info.get("trailingAnnualDividendYield"),
        'earnings_growth': info.get('earningsQuarterlyGrowth'),
        'roe': info.get('returnOnEquity'),
        'roa': info.get('returnOnAssets'),
        'debt_to_equity': info.get('debtToEquity') * 0.01,
        'pb_ratio': info.get('priceToBook'),
        'free_cash_flow': info.get('freeCashflow'),
        'market_cap': info.get('marketCap'),
        'rev_growth': info.get("revenueGrowth"),
        'sector': info.get('sector')
    }
    if data["peg_ratio"] and data["peg_ratio"] > 0 and data["pe_ratio"]:
        data["growth"] = data["pe_ratio"] / data["peg_ratio"]
    else:
        data["growth"] = 0
    return data
stock_data("GOOGL")

# %%
def peter_lynch(ticker):
    data = stock_data(ticker)

    growth = data.get("growth")
    dividend_yield = data.get("dividend_yield%") or 0
    pe_ratio = data.get("pe_ratio")

    score = ((growth*2 + dividend_yield)/pe_ratio) if growth is not None and pe_ratio is not None else 0
    #score = ((data["growth"]*2) + data["dividend_yield%"]) / data["pe_ratio"]

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
    verdict = (
    "PASSED✅"
    if (
        roe is not None and roe >= 0.12
        and de is not None and de <= 1.0
        and pe is not None and 0 < pe < 30
        and earn is not None and earn > 0
    )
    else "DID NOT PASS❌"
)

    result = []
    score = 0
    if roe is not None and roe >= 0.12:
        result.append("ROE: Good✅\n")
        score += 30
    else:
        result.append("ROE: Weak❌")

    if de is not None and de <= 1.0:
        result.append("Debt: Low✅")
        score += 20
    else:
        result.append("Debt: High❌")

    if pe is not None and 0 < pe < 30:
        result.append("PE: Fair✅")
        score += 30
    else:
        result.append("PE: High❌")

    if earn is not None and earn > 0:
        result.append("Growth: Positive✅")
        score +=20
    else:
        result.append("Growth: Negative❌")

    return f"{score}%"
    return f"{verdict} | {' | '.join(result)}"

    return f"\nWarren Buffett: {verdict} | {' | '.join(result)}"

warren_buffett("TSLA")

# %%
def philip_fisher(ticker):
    data = stock_data(ticker)
    rev_growth = data["rev_growth"]
    earn_growth = data["earnings_growth"]
    roe = data["roe"]
    score = 0

    if (earn_growth is not None and earn_growth >= 0.1) and (rev_growth is not None and rev_growth > 0.1) and (roe is not None and roe > 0.15):
        verdict = "PASSED✅"
    else:
        verdict = "DID NOT PASS❌"

    details = []
    if earn_growth is not None and earn_growth >= 0.1:
        details.append("Earnings Growth: Good✅")
        score += 35
    else:
        details.append("Earnings Growth: Weak❌")

    if rev_growth is not None and rev_growth > 0.1:
        details.append("Sales Growth: Good✅")
        score += 35
    else:
        details.append("Sales Growth: Weak❌")

    if roe is not None and roe > 0.15:
        details.append("ROE: Strong✅")
        score +=30
    else:
        details.append("ROE: Weak❌")

    result = (
        f"\nPhilip Fisher: {verdict} | {' | '.join(details)}"
    )
    return f"{score}%"
    return result

philip_fisher("AMZN")

# %%
def magic_formula(ticker):
    data = stock_data(ticker)

    ebit_ttm = data.get("ebit_ttm")
    ev = data.get("enterprise_value")
    invested_capital = data.get("invested_capital")

    earn_yield = (ebit_ttm / ev) if ebit_ttm is not None and ev not in (None, 0) else None
    roic = data.get("ROIC")

    score = 0

    if (earn_yield is not None and earn_yield > 0.07) and (roic is not None and roic > 0.1):
        verdict = "PASSED✅"
    else:
        verdict = "DID NOT PASS❌"

    details = []
    if earn_yield is not None and earn_yield > 0.07:
        details.append(f"Earnings Yield: Good✅ ({earn_yield:.3f})")
        score += 50
    else:
        details.append(f"Earnings Yield: Weak❌ ({earn_yield:.3f})" if earn_yield is not None else "Earnings Yield: N/A❌")

    if roic is not None and roic > 0.1:
        details.append(f"ROIC: Strong✅ ({roic:.3f})")
        score += 50
    else:
        details.append(f"ROIC: Weak❌ ({roic:.3f})" if roic is not None else "ROIC: N/A❌")

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
    
    if (pb is not None and pb <= 2) and (free_cash_flow is not None and free_cash_flow > 0) and (de is not None and de <= 0.5):
        verdict = "PASSED ✅"
    else:
        verdict = "DID NOT PASS ❌"

    details = []
    if pb is not None and pb <= 2:
        details.append(f"P/B: Low✅ ({pb:.2f})")
        score += 34
    else:
        details.append(f"P/B: High❌ ({pb:.2f})" if pb is not None else "P/B: N/A❌")

    if free_cash_flow is not None and free_cash_flow > 0:
        details.append(f"FCF: Positive✅ ({free_cash_flow:.0f})")
        score += 33
    else:
        details.append(f"FCF: Negative❌ ({free_cash_flow:.0f})" if free_cash_flow is not None else "FCF: N/A❌")

    if de is not None and de <= 0.5:
        details.append(f"Debt/Equity: Low✅ ({de:.2f})")
        score += 33
    else:
        details.append(f"Debt/Equity: High❌ ({de:.2f})" if de is not None else "Debt/Equity: N/A❌")

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
        (pe is not None and pe <= 20.0)
        and (pb is not None and pb <= 3.0)
        and (de is not None and pb <= 1.0)
        and (div_yield is not None and div_yield > 0)
        and (earn_growth is not None and earn_growth > 0)
    ):
        verdict = "PASSED ✅"
    else:
        verdict = "DID NOT PASS ❌"

    details = []

    if pe is not None and pe <= 20.0:
        details.append(f"P/E: Good✅ ({pe:.2f})")
        score += 20
    else:
        details.append(f"P/E: High❌ ({pe:.2f})" if pe is not None else "P/E: N/A❌")

    if pb is not None and pb <= 3.0:
        details.append(f"P/B: Good✅ ({pb:.2f})")
        score += 20
    else:
        details.append(f"P/B: High❌ ({pb:.2f})" if pb is not None else "P/B: N/A❌")

    if de is not None and de <= 1.0:
        details.append(f"D/E: Good✅ ({de:.2f})")
        score += 20
    else:
        details.append(f"D/E: High❌ ({de:.2f})" if de is not None else "D/E: N/A❌")

    if div_yield is not None and div_yield > 0:
        details.append(f"Dividend: Present✅ ({div_yield:.2f}%)")
        score += 20
    else:
        details.append(f"Dividend: None❌ ({div_yield:.2f}%)" if div_yield is not None else "Dividend: N/A❌")

    if earn_growth is not None and earn_growth > 0:
        details.append(f"Earnings Growth: Positive✅ ({earn_growth:.2f})")
        score += 20
    else:
        details.append(f"Earnings Growth: Negative❌ ({earn_growth:.2f})" if earn_growth is not None else "Earnings Growth: N/A❌")

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
    roic = data["ROIC"]
    details = []
    score = 0

    if roe is not None and roe >= 0.14:
        score += 12.5
        details.append(f"ROE: {roe:.2f} ✅")
    else:
        details.append(f"ROE: {roe:.2f} ❌" if roe is not None else "ROE: N/A ❌")

    if roic is not None and roic >= 0.10:
        score += 12.5
        details.append(f"ROA: {roic:.2f} ✅")
    else:
        details.append(f"ROA: {roic:.2f} ❌" if roic is not None else "ROA: N/A ❌")

    if earnings_growth is not None and earnings_growth >= 0.12:
        score += 12.5
        details.append(f"Earnings Growth: {earnings_growth:.2f} ✅")
    else:
        details.append(f"Earnings Growth: {earnings_growth:.2f} ❌" if earnings_growth is not None else "Earnings Growth: N/A ❌")

    if revenue_growth is not None and revenue_growth >= 0.10:
        score += 12.5
        details.append(f"Revenue Growth: {revenue_growth:.2f} ✅")
    else:
        details.append(f"Revenue Growth: {revenue_growth:.2f} ❌" if revenue_growth is not None else "Revenue Growth: N/A ❌")

    if pe is not None and pe <= 40:
        score += 12.5
        details.append(f"P/E: {pe:.2f} ✅")
    else:
        details.append(f"P/E: {pe:.2f} ❌" if pe is not None else "P/E: N/A ❌")

    if peg is not None and peg < 2.5:
        score += 12.5
        details.append(f"PEG: {peg:.2f} ✅")
    else:
        details.append(f"PEG: {peg:.2f} ❌" if peg is not None else "PEG: N/A ❌")

    if de is not None and de <= 0.5:
        score += 12.5
        details.append(f"D/E: {de:.2f} ✅")
    else:
        details.append(f"D/E: {de:.2f} ❌" if de is not None else "D/E: N/A ❌")

    if fcf is not None and fcf > 0:
        score += 12.5
        details.append(f"FCF: {fcf:.0f} ✅")
    else:
        details.append(f"FCF: {fcf:.0f} ❌" if fcf is not None else "FCF: N/A ❌")
        
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
            "Dividend Yield %": data["dividend_yield%"] * 100,
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

        rows.append(row)

    df = pd.DataFrame(rows)

    # convert Peter Lynch column to float
    PL = df["Peter Lynch"].astype(float)

    # calculate PL_Score for each row
    PL_Score = []
    for val in PL:
        if 0 < val < 0.50:
            PL_Score.append(0)
        elif 0.50 <= val < 1.00:
            PL_Score.append(20)
        elif 1.00 <= val < 1.50:
            PL_Score.append(50)
        elif 1.50 <= val < 2.00:
            PL_Score.append(75)
        elif 2.00 <= val < 2.50:
            PL_Score.append(100)
        elif val >= 2.50:
            PL_Score.append(100)
        else:
            PL_Score.append(0)

    # attach PL_Score column to DataFrame

    # last 6 columns (method outputs)
    df_last6 = df.iloc[:, -6:].replace("%", "", regex=True).astype(float)

    avg_score = df_last6.mean(axis=1)
    avg_score = ((avg_score * 6) + PL_Score) / 7
    df["Final Score"] = avg_score.round(2).astype(str) + "%"
    df = df.round(2)

    return df


# %%
pd.options.display.max_columns = None
df = stock_df(["GOOGL", "ASML", "TSLA", "MSFT", "AMZN", "AAPL"])
df.head()


