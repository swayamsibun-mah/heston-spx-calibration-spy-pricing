"""
===========================================================
Filtering options data appropriate for model calibration 
===========================================================

"""

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Heston calibration filter
# ─────────────────────────────────────────────────────────────────────────────

# ── Heston filter thresholds — edit here to tune ─────────────────────────────
HESTON_PARAMS = {
    #-------------------------------------------------------------------------------
    # DTE: 30–300 days. <30 = gamma noise; >250 = low liquidity & rate sensitivity
    #-------------------------------------------------------------------------------
    "dte_min":           30,
    "dte_max":           300,
    #-------------------------------------------------------------------------------
    # Moneyness: 0.85–1.15 (±15%). Beyond this, prices are illiquid & IV unreliable
    #-------------------------------------------------------------------------------
    "moneyness_min":     0.80,
    "moneyness_max":     1.25,
    #---------------------------------------------------------------------------
    # Bid must be strictly positive — zero bid = no real market maker interest
    #---------------------------------------------------------------------------
    "bid_min":           0.0,
    #----------------------------------------------------------------------------
    # Bid-ask spread ≤ 15% of mid. Tighter than before to weed out stale quotes
    #----------------------------------------------------------------------------
    "spread_pct_max":    0.15,
    #---------------------------------------------------------------------------
    # Minimum price of $0.50 — below this, quotes are unreliable penny options
    #---------------------------------------------------------------------------
    "price_min":         0.50,
    #------------
    # Liquidity
    #------------
    "volume_min":        10,
    "open_interest_min": 100,
    #----------------------------------------------------------------
    # IV sanity: 5%–150%. Heston cannot fit extreme IVs meaningfully
    #----------------------------------------------------------------
    "iv_min":            0.05,
    "iv_max":            1.50,
}


def heston_filter(df: pd.DataFrame, OTM: bool = True, params: dict = None) -> pd.DataFrame:
    """
    Select 80–100 options appropriate for Heston model calibration.

    Design principles
    -----------------
    1. OTM only — calls where K >= spot, puts where K < spot.
       One price per strike avoids put-call parity redundancy.
    2. DTE 30–250 days — captures enough term structure for Heston's
       mean-reversion (kappa) and vol-of-vol (xi) without going into
       illiquid long-dated territory.
    3. Moneyness 0.85–1.15 — the liquid, informative part of the smile.
       Deep OTM options have unreliable IVs and near-zero vegas.
    4. Hard liquidity gates: positive bid, tight spread, min price $0.50.
    5. Per-expiry cap of 20 contracts, chosen to maximise moneyness
       coverage while keeping the surface balanced across maturities.

    Parameters
    ----------
    df     : raw options DataFrame from get_options_df()
    params : dict to override any value in HESTON_PARAMS

    Returns
    -------
    Filtered DataFrame (80–100 rows) with extra columns:
        moneyness   = strike / spot
        spread_pct  = (ask - bid) / mid   [liquidity quality indicator]
    """
    p = {**HESTON_PARAMS, **(params or {})}
    df = df.copy()

    def _n(label):
        print(f"    [{len(df):>4}]  {label}")

    print("\n  Applying Heston filters …")
    #--------------------------
    # Step 1: derived columns
    #--------------------------
    df["moneyness"] = (df["strike"] / df["spot"]).round(6)
    bid = pd.to_numeric(df["bid"],      errors="coerce")
    ask = pd.to_numeric(df["ask"],      errors="coerce")
    mid = pd.to_numeric(df["midPrice"], errors="coerce")
    df["spread_pct"] = ((ask - bid) / mid.where(mid > 0)).round(4)
    _n(f"start")
    #---------------------
    # Step 2: DTE window
    #---------------------
    df = df[(df["dte"] >= p["dte_min"]) & (df["dte"] <= p["dte_max"])].copy()
    _n(f"DTE {p['dte_min']}–{p['dte_max']} days")
    if df.empty:
        print("  WARNING: no contracts survive DTE filter.")
        return df
    #-------------------------
    # Step 3: moneyness band
    #-------------------------
    df = df[(df["moneyness"] >= p["moneyness_min"]) &
            (df["moneyness"] <= p["moneyness_max"])].reset_index(drop=True)
    _n(f"moneyness {p['moneyness_min']}–{p['moneyness_max']}")
    if df.empty:
        print("  WARNING: no contracts survive moneyness filter.")
        return df
    #-----------------------------------------
    # Step 4: OTM only — one side per strike
    #-----------------------------------------
    if OTM:
        spot_s = df["spot"].values
        call_mask = (df["type"] == "call").values & (df["strike"].values >= spot_s)
        put_mask  = (df["type"] == "put").values  & (df["strike"].values <  spot_s)
        df = df[call_mask | put_mask].reset_index(drop=True)
        _n("OTM only (calls K≥spot, puts K<spot)")
        if df.empty:
            print("  WARNING: no contracts survive OTM filter.")
            return df
    #--------------------------------
    # Step 5: bid strictly positive
    #-----------------------------------------
    df = df[pd.to_numeric(df["bid"], errors="coerce") > p["bid_min"]].copy()
    _n(f"bid > {p['bid_min']}")
    #-----------------------------
    # Step 6: minimum mid-price
    #-----------------------------
    df = df[pd.to_numeric(df["midPrice"], errors="coerce") >= p["price_min"]].copy()
    _n(f"midPrice ≥ ${p['price_min']}")
    #-------------------------------------------------------------
    # Step 7: spread filter (skip rows where spread_pct is NaN)
    #-------------------------------------------------------------
    spread_ok = df["spread_pct"].isna() | (df["spread_pct"] <= p["spread_pct_max"])
    df = df[spread_ok].copy()
    _n(f"spread ≤ {p['spread_pct_max']*100:.0f}% of mid")
    #--------------------------------------------------------
    # Step 8: volume (only if column present and non-empty)
    #--------------------------------------------------------
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        if vol.notna().any():
            df = df[vol >= p["volume_min"]].copy()
            _n(f"volume ≥ {p['volume_min']}")
    #---------------------------------------------------------------
    # Step 9: open interest (only if column present and non-empty)
    #---------------------------------------------------------------
    if "openInterest" in df.columns:
        oi = pd.to_numeric(df["openInterest"], errors="coerce")
        if oi.notna().any():
            df = df[oi >= p["open_interest_min"]].copy()
            _n(f"openInterest ≥ {p['open_interest_min']}")
    #----------------------------
    # Step 10: IV sanity bounds
    #----------------------------
    iv = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    df = df[(iv >= p["iv_min"]) & (iv <= p["iv_max"])].copy()
    _n(f"IV {p['iv_min']*100:.0f}%–{p['iv_max']*100:.0f}%")

    if df.empty:
        print("  WARNING: all contracts filtered out. "
              "Relax HESTON_PARAMS thresholds and retry.")
        return df
    #-----------------------------------------
    # Step 11: final sort + column ordering
    #-----------------------------------------
    df = df.sort_values(["dte", "moneyness"]).reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in ("moneyness", "spread_pct")]
    if "impliedVolatility" in base_cols:
        pos = base_cols.index("impliedVolatility") + 1
        ordered = base_cols[:pos] + ["moneyness", "spread_pct"] + base_cols[pos:]
    else:
        ordered = base_cols + ["moneyness", "spread_pct"]
    df = df[[c for c in ordered if c in df.columns]]
    #-------------------------
    # Step 12: summary report
    #-------------------------
    print(f"\n  {'='*56}")
    print(f"  Heston calibration set — {len(df)} contracts")
    print(f"  {'='*56}")
    print(f"  {'DTE':>6}  {'Puts':>5}  {'Calls':>5}  {'Total':>5}    Moneyness range")
    print(f"  {'-'*56}")
    for dte_val, grp in df.groupby("dte"):
        n_c  = int((grp["type"] == "call").sum())
        n_p  = int((grp["type"] == "put").sum())
        mmin = grp["moneyness"].min()
        mmax = grp["moneyness"].max()
        print(f"  {int(dte_val):>6}  {n_p:>5}  {n_c:>5}  {len(grp):>5}"
              f"    {mmin:.3f} – {mmax:.3f}")
    print(f"  {'-'*56}")
    print(f"  {'TOTAL':>6}  "
          f"{int((df['type']=='put').sum()):>5}  "
          f"{int((df['type']=='call').sum()):>5}  "
          f"{len(df):>5}\n")
    return df