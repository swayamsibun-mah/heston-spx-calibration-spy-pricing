"""
Yahoo Finance Options Data Fetcher
====================================
Fetches the full options chain for any ticker (default: ^SPX) listed in 
Yahoo finance and returns everything as a single, flat pandas DataFrame.  
A CSV file is always saved automatically in Excel for calibration work.

Columns in the output DataFrame
────────────────────────────────
  ticker            – underlying symbol (e.g. "^SPX")
  spot              – underlying price at time of fetch
  type              – "call" or "put"
  expiration        – expiry date (YYYY-MM-DD)
  dte               – calendar days to expiration
  contractSymbol    – OCC contract identifier
  strike            – strike price
  lastPrice         – last traded premium
  bid               – best bid
  ask               – best ask
  midPrice          – (bid + ask) / 2
  change            – day's price change
  percentChange     – day's % change
  volume            – contracts traded today
  openInterest      – total open contracts
  impliedVolatility – IV (as a decimal, e.g. 0.20 = 20 %)
  inTheMoney        – True / False

Jupyter usage
─────────────
    from qqq_options import get_options_df, save_csv

    df = get_options_df("QQQ")          # full chain, all expiries
    df_calls = df[df["type"] == "call"]
    df_exp   = df[df["expiration"] == df["expiration"].min()]  # nearest expiry

    save_csv(df)                        # saves qqq_options.csv in working dir

CLI usage
─────────
    python qqq_options.py                # fetch + save CSV
    python qqq_options.py --ticker SPY
    python qqq_options.py --demo         # offline / synthetic data
    python qqq_options.py --out my_file.csv
"""

import argparse
import datetime
import sys
from typing import Optional

import pandas as pd
import numpy as np
import requests

#=============================================
# Column order for the final DataFrame / CSV
#=============================================
COLUMNS = [
    "ticker",
    "spot",
    "type",
    "dte",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "midPrice",
    "volume",
    "impliedVolatility",
    "inTheMoney",
    "dividendRate",
    "dividendYield",
    "divs_before_expiry",
    "pv_dividends",
    "spot_adj",
]


#=======================
# Helpers
#=======================

def _dte(exp_str: str) -> int:
    """Calendar days from today to expiry date string (YYYY-MM-DD)."""
    today = datetime.date.today()
    exp   = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
    return (exp - today).days


def _add_meta(df: pd.DataFrame, option_type: str,
              ticker: str, spot: float, exp: str) -> pd.DataFrame:
    """Attach ticker, spot, type, expiration, dte, and midPrice columns."""
    df = df.copy()
    df["ticker"]     = ticker
    df["spot"]       = spot
    df["type"]       = option_type          # "call" or "put"
    df["expiration"] = exp
    df["dte"]        = _dte(exp)
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    df["midPrice"]   = ((bid + ask) / 2).round(4)
    return df


def _select_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only COLUMNS that exist, in order; sort by expiration then strike."""
    cols = [c for c in COLUMNS if c in df.columns]
    return (df[cols].sort_values(["dte", "strike"]).reset_index(drop=True))


def _fetch_dividend_schedule(ticker: str) -> list:
    """
    Return a list of (ex_date, amount) tuples for upcoming dividends.

    Works for both ETFs and individual stocks by using only t.dividends — 
    the historical price series endpoint — which Yahoo supports for all 
    securities. Avoids t.info and t.calendar which return HTTP 404 for ETFs
    ("No fundamentals data found for symbol").

    Strategy:
      - Fetch last 2 years of historical dividend payments via t.dividends
      - Use the last 4 payments to estimate average cadence (days) and amount
      - Project forward from the last known payment date up to 2 years out

    Returns list of (datetime.date, float) sorted ascending by date.
    """
    try:
        import yfinance as yf
        t       = yf.Ticker(ticker)
        today   = datetime.date.today()
        horizon = today + datetime.timedelta(days=730)
        divs    = []

        hist = t.dividends
        if hist is None or len(hist) < 2:
            print(f"  [dividends] no history found for {ticker} — assuming no dividends.")
            return []

        hist    = hist.sort_index()
        recent  = hist.iloc[-4:]      # last 4 payments to estimate cadence
        amounts = recent.values.tolist()
        dates   = [d.date() if hasattr(d, "date") else d for d in recent.index]

        # Average gap between payments (days) — handles monthly/quarterly/annual
        gaps    = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_gap = int(sum(gaps) / len(gaps))
        avg_amt = round(sum(amounts) / len(amounts), 4)

        # Project forward from last known ex-date
        projected = dates[-1]
        while projected <= horizon:
            projected = projected + datetime.timedelta(days=avg_gap)
            if projected >= today:
                divs.append((projected, avg_amt))

        print(f"  [dividends] {ticker}: avg ${avg_amt}/payment every ~{avg_gap} days "
              f"→ {len(divs)} projected payments through {horizon}")
        return sorted(divs, key=lambda x: x[0])

    except Exception as exc:
        print(f"  [dividends] could not fetch schedule for {ticker}: {exc}")
        return []


def _fetch_dividend_info(ticker: str) -> dict:
    """
    Return annualised dividend rate ($/share/year) and yield (decimal).
    Derived entirely from t.dividends (works for ETFs and stocks alike).
    Falls back to 0.0 / NaN gracefully for non-dividend payers.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        hist = t.dividends
        if hist is None or len(hist) == 0:
            return {"dividendRate": 0.0, "dividendYield": float("nan")}

        hist    = hist.sort_index()
        recent  = hist.iloc[-4:]
        amounts = recent.values.tolist()
        dates   = [d.date() if hasattr(d, "date") else d for d in recent.index]

        # Annualise: sum last 4 payments scaled to a year
        gaps      = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_gap   = sum(gaps) / len(gaps) if gaps else 91.0
        per_year  = 365.0 / avg_gap           # e.g. ~4 for quarterly
        annual_rate = round(sum(amounts) / len(amounts) * per_year, 4)

        # Yield = annual rate / current spot price
        spot = getattr(t.fast_info, "last_price", None) or 0.0
        yield_ = round(annual_rate / spot, 6) if spot > 0 else float("nan")

        return {"dividendRate": annual_rate, "dividendYield": yield_}

    except Exception:
        return {"dividendRate": 0.0, "dividendYield": float("nan")}


# ─────────────────────────────────────────────────────────────────────────────
# 1. yfinance  (primary source)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yfinance(ticker: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        t           = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        spot = getattr(t.fast_info, "last_price", None) or 0.0

        frames = []
        for exp in expirations:
            chain = t.option_chain(exp)
            for side, raw in (("call", chain.calls), ("put", chain.puts)):
                frames.append(_add_meta(raw, side, ticker, spot, exp))

        df  = pd.concat(frames, ignore_index=True)
        div = _fetch_dividend_info(ticker)
        df["dividendRate"]  = div["dividendRate"]
        df["dividendYield"] = div["dividendYield"]
        #df = _attach_dividend_cols(df, schedule)
        return df

    except Exception as exc:
        print(f"  [yfinance] failed: {exc}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Yahoo Finance REST API  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":  "application/json",
    "Referer": "https://finance.yahoo.com",
}


def _yf_rest(ticker: str, date_ts: int = None):
    url    = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
    params = {"date": date_ts} if date_ts else {}
    r      = requests.get(url, headers=_YF_HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _fetch_rest(ticker: str) -> Optional[pd.DataFrame]:
    try:
        root   = _yf_rest(ticker)
        result = root["optionChain"]["result"][0]
        exp_ts = result.get("expirationDates", [])
        spot   = result.get("quote", {}).get("regularMarketPrice", 0.0)

        frames = []
        for ts in exp_ts:
            exp_str = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            data    = _yf_rest(ticker, ts)
            opts    = data["optionChain"]["result"][0]["options"][0]
            for side, key in (("call", "calls"), ("put", "puts")):
                contracts = opts.get(key, [])
                if not contracts:
                    continue
                raw = pd.DataFrame([{
                    "contractSymbol":    c.get("contractSymbol"),
                    "strike":            c.get("strike"),
                    "lastPrice":         c.get("lastPrice"),
                    "bid":               c.get("bid"),
                    "ask":               c.get("ask"),
                    "change":            c.get("change"),
                    "percentChange":     c.get("percentChange"),
                    "volume":            c.get("volume"),
                    "openInterest":      c.get("openInterest"),
                    "impliedVolatility": c.get("impliedVolatility"),
                    "inTheMoney":        c.get("inTheMoney"),
                } for c in contracts])
                frames.append(_add_meta(raw, side, ticker, spot, exp_str))

        return pd.concat(frames, ignore_index=True) if frames else None

    except Exception as exc:
        print(f"  [yahoo_rest] failed: {exc}", file=sys.stderr)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_options_df(ticker: str = "^SPX", demo: bool = False) -> pd.DataFrame:
    """
    Fetch the full options chain and return it as a single flat DataFrame.

    Parameters
    ----------
    ticker : str   Underlying symbol, e.g. "^SPX", "SPY", "QQQ", "AAPL".
    demo   : bool  If True, return synthetic data (no network required).

    Returns
    -------
    pd.DataFrame with columns defined in COLUMNS (see module docstring).
    """
    if demo:
        print("  [demo] generating synthetic options data …")
        return _select_cols(_fetch_demo(ticker))

    print(f"Fetching options data for {ticker} …")

    print("  Trying yfinance …")
    df = _fetch_yfinance(ticker)
    if df is not None:
        print(f"  OK — {len(df):,} contracts fetched.")
        return _select_cols(df)

    print("  Trying Yahoo Finance REST API …")
    df = _fetch_rest(ticker)
    if df is not None:
        print(f"  OK — {len(df):,} contracts fetched.")
        return _select_cols(df)

    print("  All live sources failed — using demo data.")
    return _select_cols(_fetch_demo(ticker))


def save_csv(df: pd.DataFrame, path: str = "qqq_options.csv") -> str:
    """
    Save the options DataFrame to a CSV file ready to open in Excel.
    Returns the absolute path of the saved file.
    """
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df):,} rows × {len(df.columns)} columns)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Jupyter / CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _is_jupyter() -> bool:
    try:
        return get_ipython().__class__.__name__ in (   # type: ignore[name-defined]
            "ZMQInteractiveShell", "TerminalInteractiveShell")
    except NameError:
        return False


def main(ticker: str = "^SPX", demo: bool = False,
         out: str = "spx_options.csv", live: bool=False) -> pd.DataFrame:
    """
    Fetch options data, display the DataFrame, and auto-save a CSV.

    Jupyter
    -------
        from qqq_options import main
        df = main()               # fetch QQQ, display df, save qqq_options.csv
        df = main("SPY")          # different ticker → saves spy_options.csv
        df = main(demo=True)      # offline synthetic data

    CLI
    ---
        python spx_options.py
        python spx_options.py --ticker ^SPX
        python spx_options.py --demo
        python spx_options.py --out my_data.csv
    """
    pd.set_option("display.max_columns",  20)
    pd.set_option("display.width",       160)
    pd.set_option("display.max_rows",     30)
    pd.set_option("display.float_format", "{:.4f}".format)

    if not _is_jupyter():
        parser = argparse.ArgumentParser(description="Fetch options chain data")
        parser.add_argument("--ticker", default=ticker,
                            help="Ticker symbol (default: ^SPX)")
        parser.add_argument("--demo",   action="store_true", default=demo,
                            help="Use synthetic demo data (no network needed)")
        parser.add_argument("--out",    default=out,
                            help="Output CSV path (default: spx_options.csv)")
        args, _ = parser.parse_known_args()
        ticker, demo, out = args.ticker, args.demo, args.out
    
    schedule = _fetch_dividend_schedule(ticker)
    out = f"../options_data/{ticker.lower()}_options.csv"
    out_div = f"../options_data/{ticker.lower()}_div_options"
    
    #-------------------------------------
    # Auto-name the CSV after the ticker
    #-------------------------------------
    if live:
        df = get_options_df(ticker=ticker, demo=demo)
        save_csv(df, out)
        np.save(out_div, np.array(schedule))
    else:
        df = pd.read_csv(out)
        schedule = np.load("".join([out_div, ".npy"]), allow_pickle=True)
        return df, schedule
    # ── Summary header ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {ticker.upper()} Options Chain")
    print(f"{'═'*60}")
    print(f"  Rows       : {len(df):,}")
    print(f"  Expiries   : {df['dte'].nunique()}  "
          f"(dte {df['dte'].min()} → {df['dte'].max()} days)")
    print(f"  Strikes    : {df['strike'].nunique()}  "
          f"(${df['strike'].min():g} – ${df['strike'].max():g})")
    print(f"{'═'*60}\n")

    # ── Save full chain CSV ──────────────────────────────────────────────────
    print()
    return df, schedule