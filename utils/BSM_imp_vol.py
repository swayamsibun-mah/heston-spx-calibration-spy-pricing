import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(option_type, S, K, T, r, q, sigma):
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    option_type : str
        'call' or 'put'
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Continuously compounded risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        elif option_type.lower() == 'put':
            return max(K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    d1 = (np.log(S / K)+ (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1)- K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2)- S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


def implied_volatility(option_type, market_price, S, K, T, r, q=0.0, tol=1e-8):
    """
    Compute implied volatility using Brent's method.

    Returns
    -------
    float
        Implied volatility
    """
    #-------------------------
    # Check arbitrage bounds
    #-------------------------
    if option_type.lower() == 'call':
        lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S * np.exp(-q * T)
    elif option_type.lower() == 'put':
        lower = max(K * np.exp(-r * T) - S * np.exp(-q * T),0.0)
        upper = K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    if market_price < lower or market_price > upper:
        raise ValueError(f"Market price {market_price:.4f} violates "f"arbitrage bounds [{lower:.4f}, {upper:.4f}]")
    def objective(sigma):
        return (black_scholes_price(option_type, S, K, T, r, q, sigma)- market_price)
    try:
        iv = brentq(objective,a=1e-4, b=5.0,xtol=tol) # Search between 0.01% and 500% volatility
    except ValueError:
        iv = 0.
    return iv