import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# User Inputs
# =====================================================

SERIES = {"SOFR": 1/365}

# =====================================================
# Download latest observation for a FRED series
# =====================================================

def get_latest_fred_value(series_id, api_key):
    url = ("https://api.stlouisfed.org/fred/series/observations")
    params = {"series_id": series_id, "api_key": api_key,
              "file_type": "json", "sort_order": "desc",
              "limit": 1}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs = data["observations"][0]
    return float(obs["value"])/100.0

# =====================================================
# Get the current SOFR rate
# =====================================================

def get_sofr_rate(api_key):
    maturities = []
    rates = []
    for series, maturity in SERIES.items():
        rate = get_latest_fred_value(series, api_key)
        maturities.append(maturity)
        rates.append(rate)
        print(f"{series:<15}"
              f"T={maturity:.4f} "
              f"Rate={100*rate:.3f}%")
    rate = np.array(rates)[0]
    np.save('../options_data/SOFR_rate', rate)
    return rate