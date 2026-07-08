"""
Pricing American options using QuantLib finite difference pricing engine
"""

import math
import warnings
import numpy as np
import pandas as pd
from datetime import date as pydate
import QuantLib as ql

warnings.filterwarnings('ignore')


def build_heston_engine(
    S: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma: float, rho: float,
    grid_t: int = 100, grid_x: int = 200, grid_v: int = 50,
    ) -> tuple:
    """
    Build and return (HestonModel, FdHestonVanillaEngine).
    -----------------------------------------------------------------------------
    This implementation treats dividends as continuous cash payments and
    uses the Crank-Nicholson finite-difference scheme for option pricing.
    -----------------------------------------------------------------------------
    """
    today    = ql.Settings.instance().evaluationDate
    spot_h   = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rate_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, float(r), ql.Actual365Fixed()))
    div_ts   = ql.YieldTermStructureHandle(ql.FlatForward(today, float(q), ql.Actual365Fixed()))
    process  = ql.HestonProcess(rate_ts, div_ts, spot_h, float(v0),
                                float(kappa), float(theta), float(sigma), float(rho))
    model    = ql.HestonModel(process)
    engine   = ql.FdHestonVanillaEngine(model, grid_t, grid_x, grid_v)
    return model, engine

def build_hestonDD_engine(
    S: float, r: float, K: float, dte: int, q: float,
    v0: float, kappa: float, theta: float, sigma: float, rho: float,
    dividend_dates, dividend_amounts,
    grid_t: int = 100, grid_x: int = 200, grid_v: int = 50,
    ) -> tuple:
    """
    Build and returns (HestonModel, FdHestonVanillaEngine).
    -----------------------------------------------------------------------------
    This implementation treats dividends as discrete cash payments and
    uses the Hundsdorfer-Verwer ADI finite-difference scheme for option pricing.

    Non-uniform spatial grids are employed, with mesh points concentrated
    near the strike price to improve pricing accuracy around the region
    of greatest sensitivity.

    The parameter, c_density, controls the degree of clustering:
    - c_density = 0.1 approximately corresponds to uniform spacing,
    - smaller values increase the concentration of grid points near
      the strike price.
    -----------------------------------------------------------------------------
    """
    today    = ql.Settings.instance().evaluationDate
    dividend_dates = [today + div_date for div_date in dividend_dates]
    spot_h   = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rate_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, float(r), ql.Actual365Fixed()))
    div_ts   = ql.YieldTermStructureHandle(ql.FlatForward(today, float(0.), ql.Actual365Fixed()))    
    process  = ql.HestonProcess(rate_ts, div_ts, spot_h, float(v0),
                                float(kappa), float(theta), float(sigma), float(rho))
    model    = ql.HestonModel(process)
    dummy_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, ql.Actual365Fixed()))
    bs_process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(float(S))), div_ts, rate_ts, dummy_vol)
    #-----------------------------------------------
    # Minimum and maximum of the asset price grid
    #-----------------------------------------------
    x_min = 0.75 * K
    x_max = 1.33 * K 
    c_density = 0.05 
    #-----------------
    # Variance Mesher
    #-----------------
    v_mesher = ql.FdmHestonVarianceMesher(grid_v, process, float(dte)/365.)
    #---------------------------
    # Custom Asset Price Mesher
    #---------------------------
    x_mesher = ql.FdmBlackScholesMesher(grid_x, bs_process, float(dte)/365., K, x_min, x_max, c_density)
    #------------------------------------------------------
    # Create the multi-dimensional Composite Mesher Layout
    #------------------------------------------------------
    mesher_layout = ql.FdmMesherComposite(x_mesher, v_mesher)
    #-------------------------------------------------------------------------------------
    # Declare Custom Boundary conditions or Engine layouts explicitly via Custom Instance
    #-------------------------------------------------------------------------------------
    equity_bounds = ql.FdmBoundaryConditionSet()
    adi_scheme = ql.FdmSchemeDesc.Hundsdorfer() 
    dividend_schedule = ql.DividendSchedule()
    for d, a in zip(dividend_dates, dividend_amounts):
        dividend_schedule.append(ql.FixedDividend(a, d))
    engine = ql.FdHestonVanillaEngine(model, dividend_schedule,
                                      grid_t, grid_x, grid_v,
                                      2, adi_scheme)
    return model, engine


def price_american_option(
    engine,
    K: float,
    days: int,
    option_type: str = 'put',   # 'call' or 'put'
    ) -> dict:
    """
    Price American option(s) with a pre-built FD Heston engine.

    Returns dict with keys 'call' and/or 'put', each containing:
    price, delta, gamma, theta (theta is per calendar day).
    """
    today    = ql.Settings.instance().evaluationDate
    maturity = today + int(days)
    exercise = ql.AmericanExercise(today, maturity)
    results  = {}
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K)) if option_type=='put' \
             else ql.PlainVanillaPayoff(ql.Option.Call, float(K))
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    results = {'price': round(option.NPV(), 6),
               'delta': round(option.delta(), 6)}
    return results

def price_heston_american(
    S: float, K:float, r:float, q:float, days: int,                 # Option specification
    v0: float, kappa: float, theta:float, sigma: float, rho: float, # Heston parameters
    otype: str ='put',                                              # Option type and dividend type
    dividend_dates: list=[], dividend_amounts:list=[],              # List of ex-dividend dates and corresponding dividend amounts
    grid_t: int=100, grid_x: int=200, grid_v: int=50,               # Grid specifications
) -> dict:
    """Convenience wrapper: build engine + price in one call.
       ========================================================
       Parameters
       ========================================================
       discrete_div: If False, dividends are treated as 
       continuous payments through spot price adjustment. If 
       True, dividends are treated explicitly as discrete cash
       payments with jump conditions applied at the
       ex-dividend dates.
       ========================================================
       Returns
       ========================================================
       Price of the American option.
    
    """
    if dividend_dates:
        _, engine = build_hestonDD_engine(S, r, K, days, 0., v0, kappa, theta, sigma, rho, dividend_dates, dividend_amounts,
                                     grid_t, grid_x, grid_v)
    else:
        _, engine = build_heston_engine(S, r, q, v0, kappa, theta, sigma, rho,
                                     grid_t, grid_x, grid_v)
    return price_american_option(engine, K, days, otype)


def feller_condition(kappa, theta, sigma) -> tuple[float, bool]:
    """Returns (2κθ - σ², satisfied)."""
    val = 2 * kappa * theta - sigma ** 2
    return val, val > 0


def implied_vol_from_variance(v: float) -> str:
    """Convert variance to percentage vol string."""
    return f'{math.sqrt(max(v, 0)) * 100:.2f}%'