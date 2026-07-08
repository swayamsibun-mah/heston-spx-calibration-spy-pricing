"""
This module is based on the Fourier-based option pricing implementation 
using the Lewis approach presented in *Derivative Analytics with Python* by
Yves Hilpisch.
"""

import numpy as np
from scipy.integrate import quad
import warnings
warnings.simplefilter('ignore')

def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' 
    Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach.
    Parameters
    ==========
    S0: float
    initial stock/index level
    K: float
    strike price
    T: float
    time-to-maturity (for t=0)
    r: float
    constant risk-free short rate
    kappa_v: float
    mean-reversion factor
    theta_v: float
    long-run mean of variance
    sigma_v: float
    volatility of variance
    rho: float
    correlation between variance and stock/index level
    v0: float
    initial level of variance
    Returns
    =======
    call_value: float
    present value of European call option
    '''
    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0), 0, np.inf, limit=500)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    return call_value

def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' 
    Returns the integrand.
    '''
    char_func_value = H93_char_func(u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    int_func_value = 1 / (u ** 2 + 0.25) * (np.exp(1j * u * (np.log(S0 / K))) * char_func_value).real
    return int_func_value


def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' 
    Returns the characteristic function.
    '''
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v)** 2 - sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2)* ((kappa_v - rho * sigma_v * u * 1j + c2) * T - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2 * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value