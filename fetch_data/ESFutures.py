import yfinance as yf
from datetime import datetime
import pandas as pd


MONTH_CODES = {
    3: "H",   # March
    6: "M",   # June
    9: "U",   # September
    12: "Z"   # December
}


def generate_es_contracts(n_contracts=8):
    """
    Generate Yahoo Finance ES futures tickers
    e.g. ESM26.CME, ESU26.CME, ESZ26.CME ...
    """

    today = datetime.today()
    quarterly_months = [3, 6, 9, 12]
    contracts = []
    year = today.year
    while len(contracts) < n_contracts:
        for month in quarterly_months:
            expiry = datetime(year, month, 1)
            if expiry >= datetime(today.year, today.month, 1):
                code = MONTH_CODES[month]
                yy = str(year)[-2:]
                ticker = f"ES{code}{yy}.CME"
                contracts.append({"ticker": ticker,
                                  "expiry_month": month,
                                  "year": year,})
                if len(contracts) >= n_contracts:
                    break
        year += 1
    return contracts


def get_es_forward_curve(n_contracts=8):
    contracts = generate_es_contracts(n_contracts)
    rows = []
    for c in contracts:
        try:
            tk = yf.Ticker(c["ticker"])
            info = tk.fast_info
            price = info.get("lastPrice", None)
            expiry_date = pd.Timestamp(year=c["year"], month=c["expiry_month"],day=18)
            rows.append({"Ticker": c["ticker"],
                         "Expiry": expiry_date,
                         "ForwardPrice": price,})
        except Exception as e:
            expiry_date = pd.Timestamp(year=c["year"],
                                       month=c["expiry_month"],
                                       day=1)

            rows.append({"Ticker": c["ticker"],
                         "Expiry": expiry_date,
                         "ForwardPrice": None,})
    esdf = pd.DataFrame(rows) 
    esdf.to_csv(f"../options_data/esfutures.csv", index=False)
    return esdf


if __name__ == "__main__":
    curve = get_es_forward_curve(8)
    print(curve)