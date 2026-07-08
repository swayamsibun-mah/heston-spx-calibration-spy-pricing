#=======================================================================================
# This module is mainly based on the paper "ADI schemes for pricing American options   #
# under the Heston model" by Tinne Haentjens and Karel J. in ’t Hout, and adapted to   #
# handle dividend jumps on ex-dividend dates through asset-grid interpolation
#=======================================================================================

import numpy as np
import math
import copy
from scipy.linalg import solve_banded
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.sparse import block_diag
import scipy.sparse as sparse
import matplotlib.pyplot as plt

class HestonFD_ADI_HV:
    """
    American Option Pricing using the Hundsdorfer–Verwer (HV) ADI Scheme
    under the Heston stochastic volatility model.

    This class implements a finite-difference solver for American-style
    options using an Alternating Direction Implicit (ADI) scheme. The
    model supports stochastic volatility (Heston framework) and discrete
    cash dividends through spot jump conditions at ex-dividend dates.

    Parameters
    ----------
    S0 : float
        Initial spot price of the underlying asset.

    r : float
        Risk-free interest rate (continuously compounded).

    K : float
        Strike price of the option.

    T : float
        Time to maturity in years.

    otype : str
        Option type. Must be either 'put' or 'call'.

    heston_params : list of float
        Parameters of the Heston stochastic volatility model, typically
        including:
        - v0 : initial variance
        - kappa : mean reversion speed
        - theta : long-run variance
        - sigma : volatility of variance
        - rho : correlation between Brownian motions

    dividend_dates : list of float, optional
        Ex-dividend dates (in years), specified in increasing order.

    dividend_amounts : list of float, optional
        Cash dividend amounts corresponding to each date in
        `dividend_dates`.

    Attributes
    ----------
    exercise_surface : ndarray
        Stored optimal exercise surface computed during the backward
        time-stepping procedure.

    Methods
    -------
    price_heston_dense()
        Computes the option price using the HV-ADI scheme. Also stores
        the optimal exercise surface for post-processing.

    price_heston()
        Implements sparsing of tridiagonal matrices for 
        computational efficiency

    smooth_exercise_surface()
        Performs a smoothing of the exercise boundary surface through
        cubic splines.

    plot_exercise_boundary()
        Plots the optimal exercise boundary from the
        computed exercise surface.

    Notes
    -----
    - Discrete dividends are incorporated via a spot-shift jump condition:
      V(S, t-) = V(S - D, t+).
    - The solver is designed for backward time evolution in a PDE
      framework.
    - The stored exercise surface can be used to analyze the free boundary
      dynamics for American puts and American calls.
    """
    
    def __init__(self, S0, r, K, T, otype, heston_params, dividend_dates=[], dividend_amounts=[]):
        self.S0 = S0
        self.r = r
        self.K = K
        self.T = T
        self.otype = otype
        self.heston_params = heston_params
        self.dividend_dates = dividend_dates
        self.dividend_amounts = dividend_amounts

    def price_heston(self, m1 = 100, m2 = 50, M = 50, display_exercise=False):
        kappa, theta, sigma, rho, v0 = self.heston_params
        dividend_dates = np.array(self.dividend_dates)
        dividend_amounts = np.array(self.dividend_amounts[::-1])
        S0 = self.S0
        r = self.r
        K = self.K
        T = self.T 
        otype = self.otype
        div_dates = (T - dividend_dates)[::-1]
        Theta = 1/2.+np.sqrt(3.)/6.
        S_max = 14*K
        V_max = 5.
        d2 = V_max/500.
        d1 = S0/20. if otype=='put' else S0/90.
        if otype=='put':
            S_left = 0.65*min(S0, K) 
            S_right = 1.3*max(S0, K) if len(dividend_dates)>0 else 1.1*max(S0, K)
        else:
            S_left = 0.9*min(S0, K) if len(dividend_dates)>0 else 0.8*min(S0, K)
            S_right = 1.4*max(S0, K) if len(dividend_dates)>0 else 1.2*max(S0, K)

        xi_min = np.arcsinh(-S_left/ d1)
        xi_int = (S_right - S_left)/d1
        xi_max = xi_int + np.arcsinh((S_max - S_right)/d1)
        def SPhi(xi):
            """
            Creates the asset price grid
            """
            if xi_min <= xi < 0:
                return S_left + d1*np.sinh(xi)
            elif 0 <= xi <= xi_int:
                return S_left + d1 *xi
            elif xi_int < xi <= xi_max:
                return S_right + d1 * np.sinh(xi - xi_int)
            else:
                raise ValueError("Grid points should be in between 0 and S_max.")
        S_grid = np.array([SPhi(xi) for xi in np.linspace(xi_min, xi_max, m1+1)])
        v_grid = np.array([d2*np.sinh(j * 1/m2 * np.arcsinh(V_max/d2)) for j in range(m2+1)])
        t_grid = np.linspace(0., T, M+1)
        #--------------------------------------------------
        # Adding the ex-dividend dates into the time grid
        #--------------------------------------------------
        div_indices = np.searchsorted(t_grid, div_dates)
        rel_div_dates = div_dates[np.minimum(np.abs(div_dates - t_grid[div_indices]), np.abs(div_dates - t_grid[div_indices-1]))>0.0001*T/M]
        t_grid = np.sort(np.concatenate((t_grid, rel_div_dates)))
        M += len(rel_div_dates)
        idx = np.searchsorted(t_grid, div_dates)
        div_indices = np.where(np.abs(t_grid[idx-1] - div_dates) < np.abs(t_grid[idx] - div_dates), idx-1, idx)
        t_div_grid = t_grid[div_indices]
        dt_grid = t_grid[1:] - t_grid[:-1]
        #--------------------------------------------------------------------
        # Initalizing matrices associated with the differential operators
        # where only their three diagonals are used
        #--------------------------------------------------------------------
        d_ds   = np.zeros((3, m1), dtype=float)
        d2_ds2 = np.zeros((3, m1), dtype=float)
        d_dv   = np.zeros((3, m2 + 1), dtype=float)
        d2_dv2 = np.zeros((3, m2 + 1), dtype=float)
        #---------------------
        # S convection term
        #---------------------
        ds = S_grid[1:] - S_grid[:-1]
        S_grid = S_grid[1:]
        d_ds[0, :-1]  =  (np.ones(m1-1, dtype=float)/ds[1:])
        d_ds[1, :-1] = -(np.ones(m1-1, dtype=float)/ds[1:])
        #---------------------
        # S diffusion term
        #---------------------
        d2_ds2[0, :-1]  =  (np.ones(m1-1, dtype=float)/((ds[1:])*(ds[:-1] + ds[1:])))
        d2_ds2[1, :-1] = -(np.ones(m1-1, dtype=float)/(ds[:-1]*ds[1:]))
        d2_ds2[2, :-2] =  (np.ones(m1-2, dtype=float)/((ds[1:-1])*(ds[1:-1] + ds[2:])))
        d2_ds2 = 2 * d2_ds2
        #---------------------
        # v convection term
        #---------------------
        dv = v_grid[1:] - v_grid[:-1]
        idx = np.searchsorted(v_grid, theta, side='right') - 1
        d_dv[2, :idx+1]   =   (np.ones(idx+1, dtype=float)/dv[:idx+1])
        d_dv[1, :idx+1]   =  -(np.ones(idx+1, dtype=float)/dv[:idx+1])
        d_dv[1, idx+1:]   =   (np.ones(m2-idx, dtype=float)/dv[idx:])
        d_dv[0, idx:-1]   =  -(np.ones(m2-idx, dtype=float)/dv[idx:])
        #---------------------
        # v diffusion term
        #---------------------
        d2_dv2[0, :-2]  =   np.ones(m2-1, dtype=float)/((dv[:-1])*(dv[:-1] + dv[1:]))
        d2_dv2[1, 1:-1] = - np.ones(m2-1, dtype=float)/((dv[1:]*dv[:-1]))
        d2_dv2[2, 1:-1] =   np.ones(m2-1, dtype=float)/((dv[1:])*(dv[:-1] + dv[1:]))
        d2_dv2 = 2 * d2_dv2
        #-------------------------------
        # Payoff at zero asset price
        #-------------------------------
        Cs0_val = K if otype=='put' else 0. 
        #----------------------------------------------
        # Function for computing the mixed derivative
        # of the option value
        #----------------------------------------------
        def mixed_deriv(arr, Cs0_val = Cs0_val):
            """
            Computes the mixed derivative of the option value
            with respect to the variance and asset price.
            """
            deriv_arr = np.zeros_like(arr)
            b1 = -dv[1:]/(dv[:-1]*(dv[:-1] + dv[1:]))
            b2 = (dv[1:] - dv[:-1])/(dv[1:] * dv[:-1])
            b3 = dv[:-1]/(dv[1:]*(dv[:-1] + dv[1:]))
            c1 = -ds[2:]/(ds[1:-1]*(ds[1:-1] + ds[2:]))
            c2 = (ds[2:] - ds[1:-1])/(ds[1:-1]*ds[2:])
            c3 = ds[1:-1]/(ds[2:]*(ds[1:-1]+ds[2:]))
            deriv_arr[1:-1, 1:-1] = (b1[None, :]*c1[:, None]) * arr[:-2, :-2] + (b2[None, :]*c1[:, None]) * arr[:-2, 1:-1] + (b3[None, :]*c1[:, None]) * arr[:-2, 2:]\
                                  + (b1[None, :]*c2[:, None]) * arr[1:-1, :-2] + (b2[None, :]*c2[:, None]) * arr[1:-1, 1:-1] + (b3[None, :]*c2[:, None]) * arr[1:-1, 2:]\
                                  + (b1[None, :]*c3[:, None]) * arr[2:, :-2] + (b2[None, :]*c3[:, None]) * arr[2:, 1:-1] + (b3[None, :]*c3[:, None]) * arr[2:, 2:]
            deriv_arr[0, 1:-1] = b1[None, :]*(-ds[1]/(ds[0]*(ds[0]+ds[1])))*Cs0_val + b1[None, :]*(ds[1]-ds[0])/(ds[0]*ds[1])*arr[0,:-2]  + b1[None, :]*ds[0]/(ds[1]*(ds[0]+ds[1]))*arr[1,:-2]\
                                +b2[None, :]*(-ds[1]/(ds[0]*(ds[0]+ds[1])))*Cs0_val + b2[None, :]*(ds[1]-ds[0])/(ds[0]*ds[1])*arr[0,1:-1] + b2[None, :]*ds[0]/(ds[1]*(ds[0]+ds[1]))*arr[1,1:-1]\
                                +b3[None, :]*(-ds[1]/(ds[0]*(ds[0]+ds[1])))*Cs0_val + b3[None, :]*(ds[1]-ds[0])/(ds[0]*ds[1])*arr[0,2:]   + b3[None, :]*ds[0]/(ds[1]*(ds[0]+ds[1]))*arr[1,2:]
            return deriv_arr
        #------------------------------------------------------------------
        # Creates a 2*m1(m2+1) matrix for the S-direction operators
        #------------------------------------------------------------------
        As2 = np.zeros((3, m1), dtype=float)
        As1 = np.zeros((3, m1), dtype=float)
        As2[0, 1:] = 1/2*np.square(S_grid[:-1]) * d2_ds2[0, :-1]
        As2[1, :]  = 1/2*np.square(S_grid) * d2_ds2[1, :]
        As2[2, :-1] = 1/2*np.square(S_grid[1:]) * d2_ds2[2, :-1]
        As1[0, 1:] = r * S_grid[:-1] * d_ds[0, :-1]
        As1[1, :]  = r* S_grid * d_ds[1, :] - r/2.
        As1[2, :-1] = r * S_grid[1:] * d_ds[2, :-1]
        As = np.hstack([v_grid[i] * As2 + As1 for i in range(m2+1)])
        #------------------------------------------------------------------
        # Creates a 3*(m2+1) matrix for the v-direction operators
        #------------------------------------------------------------------
        Av = np.zeros((3, m2+1), dtype=float)
        Av[0, 1:]  = 1/2 * sigma**2 * v_grid[1:] * d2_dv2[0, :-1] + kappa * (theta - v_grid[1:]) * d_dv[0, :-1]
        Av[1, :]   = 1/2 * sigma**2 * v_grid * d2_dv2[1, :] + kappa * (theta - v_grid) * d_dv[1, :] - r/2.
        Av[2, :-1] = 1/2 * sigma**2 * v_grid[:-1] * d2_dv2[2, :-1] + kappa * (theta - v_grid[:-1]) * d_dv[2, :-1]
        Av_alt = np.zeros((3, m2+1), dtype=float)
        Av_alt[0, :-1] = Av[0, 1:]
        Av_alt[1: , :] = Av[1: , :]
        #----------------------
        # Initial conditions
        #----------------------
        lambb = np.zeros((m1, m2+1), dtype=float)
        #--------------------
        # Payoff at maturity
        #--------------------
        if otype == 'put':
            payoff = np.maximum(K-S_grid, 0.)
            idx = np.searchsorted(S_grid, K, side='right') - 1
            payoff[idx] = 2/(S_grid[idx+1]-S_grid[idx-1]) \
                        * (K**2/2 - K*(S_grid[idx-1]+S_grid[idx])/2 + 1/8*(S_grid[idx-1]+S_grid[idx])**2)     
        else:
            payoff = np.maximum(S_grid-K, 0.)
            idx = np.searchsorted(S_grid, K, side='right')
            payoff[idx] = 2/(S_grid[idx+1]-S_grid[idx-1]) \
                        * (K**2/2 - K*(S_grid[idx]+S_grid[idx+1])/2 + 1/8*(S_grid[idx]+S_grid[idx+1])**2)        
        U0 = np.tile(payoff[:, np.newaxis], (1, m2+1))
        Uhn1 = copy.deepcopy(U0)
        S_grid_addn = np.insert(S_grid, 0, 0.)
        #-----------------------------------------------------------------------
        # Initialising the array to store the optimal exercise boundary surface
        #-----------------------------------------------------------------------
        S_exercise = np.array([])
        #---------------------------------------------------
        # Variables to track the time-step, the dividends
        #---------------------------------------------------
        i, j, k = 0, 0, 0
        #------------------------------------------------------------------------
        # Backward time-step using the Hundsdorfer-Verwer ADI numerical scheme
        #------------------------------------------------------------------------
        #------------------    
        # Boundary terms
        #------------------
        G = 1/2*np.square(S_grid[0]) * v_grid *2*K/(ds[0]*(ds[0]+ds[1])) if otype=='put' else r*S_grid[-1]
        #================================
        # ADI-IT splitting under the
        # Hunsdorfer-Verwer scheme
        #================================
        while j<M:
            #----------------------------------------------------------------
            # Rannacher smoothing in the first two time steps
            # by fully implicit Euler method
            #----------------------------------------------------------------
            if j<2:
                dt = dt_grid[0]
                for l in range(4):                    
                    P1 = -dt/2 * As
                    P1[1, :] += 1
                    p1 = Uhn1.ravel(order='F')
                    Uint = (solve_banded((1, 1), P1, p1)).reshape((m1, m2+1), order='F')
                    P2_int = -dt*Av/2
                    P2_int[1, :] += 1
                    P2 = np.zeros((3, m2+1), dtype=float)
                    P2[0, 1:] = P2_int[2, :-1]
                    P2[1, :]  = P2_int[1, :]
                    P2[2, :-1] = P2_int[0, 1:]
                    p2 = np.transpose(Uint)
                    Uhn = np.transpose(solve_banded((1, 1), P2, p2))
                    Uhn += dt/2 * (rho*sigma*S_grid[:, None]*v_grid[None,:]*mixed_deriv(Uhn1, Cs0_val))
                    if otype=='call':
                        Uhn[-1, :] += G*dt/2
                    else:
                        Uhn[0, :] += G*dt/2
                        G *= math.exp(-r*dt/2)
                        Cs0_val *= math.exp(-r*dt/2)  
                    Uhn = np.maximum(Uhn, U0)
                    Uhn1 = Uhn
                    pass
                j=2
                continue
            dt = dt_grid[j]
            #-----------------------
            # Predictor step
            #-----------------------
            As_Uhn1 = 1/2*np.square(S_grid[:, None]) * v_grid[None,:] *(sparse.diags(d2_ds2, [1, 0, -1], shape=(m1, m1), format='csr') @ Uhn1) \
                    + r * S_grid[:, None] * np.ones(m2+1, dtype=float)[None, :] * (sparse.diags(d_ds, [1, 0, -1], shape=(m1, m1), format='csr') @ Uhn1) - r*Uhn1/2. 
            Av_Uhn1 = sparse.diags(Av_alt, [1, 0, -1], shape=(m2+1, m2+1), format='csr').T.dot(Uhn1.T).T - r * Uhn1/2.
            Y0 = Uhn1 + dt * lambb + dt * (As_Uhn1 + Av_Uhn1 + rho*sigma*S_grid[:, None] \
                                           * v_grid[None,:]*mixed_deriv(Uhn1, Cs0_val)) 
            #---------------------------
            # Adding the boundary terms
            #---------------------------
            if otype=='put':
                Y0[0, :] += dt*G
            else:
                mask = t_grid[j+1] - t_div_grid > 0
                q = r - 1/t_grid[j+1]*np.log(1 - np.dot(dividend_amounts[mask], np.exp(-r*(t_grid[j+1]-t_div_grid))[mask])/S0)
                Y0[-1, :]  += dt*G*np.exp(-q*t_grid[j+1])
            #----------------------------------------
            # Directional Implicit Corrections - 1
            #----------------------------------------
            p1 = (Y0 - Theta * dt * As_Uhn1).ravel(order='F')
            P1 = -Theta*dt*As
            P1[1, :] += 1
            Y1 = (solve_banded((1, 1), P1, p1)).reshape((m1, m2+1), order='F')
            P2_int = -Theta*dt*Av
            P2_int[1, :] += 1
            P2 = np.zeros((3, m2+1), dtype=float)
            P2[0, 1:]  = P2_int[2, :-1]
            P2[1, :]   = P2_int[1, :]
            P2[2, :-1] = P2_int[0, 1:]
            p2 = np.transpose(Y1 - Theta * dt * Av_Uhn1)
            Y2 = np.transpose(solve_banded((1, 1), P2, p2))
            #---------------------------
            # Second correction stage
            #---------------------------
            As_Y2 = 1/2*np.square(S_grid[:, None]) * v_grid[None,:] *(sparse.diags(d2_ds2, [1, 0, -1], shape=(m1, m1), format='csr') @ Y2) \
                    + r * S_grid[:, None] * np.ones(m2+1, dtype=float)[None,:] * (sparse.diags(d_ds, [1, 0, -1], shape=(m1, m1), format='csr') @ Y2) - r*Y2/2.
            Av_Y2 = sparse.diags(Av_alt, [1, 0, -1], shape=(m2+1, m2+1), format='csr').T.dot(Y2.T).T - r*Y2/2.
            Ytil0 = Y0 + 1/2 * dt * (As_Y2 - As_Uhn1 + Av_Y2 - Av_Uhn1 \
                                     + rho*sigma*S_grid[:, None] * v_grid[None,:]*(mixed_deriv(Y2, Cs0_val) - mixed_deriv(Uhn1, Cs0_val)))\
            #----------------------------------------
            # Directional Implicit Corrections - 2
            #----------------------------------------
            p1 = (Ytil0 - Theta * dt * As_Y2).ravel(order='F')
            Ytil1 = (solve_banded((1, 1), P1, p1)).reshape((m1, m2+1), order='F')
            p2 = np.transpose(Ytil1 - Theta * dt * Av_Y2)     
            Ytil2 = np.transpose(solve_banded((1, 1), P2, p2))
            #------------------------------------------------
            # Prediction update at the current time instant
            #------------------------------------------------
            Ubarn = Ytil2
            Uhn = np.maximum(Ubarn - dt * lambb, U0)
            lambb = np.maximum(0, lambb + (U0 - Ubarn)/dt)
            #---------------------------------------------------
            # Passing the current estimate of the fair-value
            # as an initial prediction for the next time step
            #---------------------------------------------------
            Uhn1 = Uhn
            #======================================
            # Spot jump on the ex-dividend date
            #======================================
            if len(div_indices)>0 and k<len(div_indices) and j == div_indices[k]-1:
                S_grid_addn = np.insert(S_grid, 0, 0.)
                Uhn_addn = np.insert(Uhn, 0, Cs0_val*np.ones(m2+1, dtype=float), axis=0)
                spline = RectBivariateSpline(S_grid_addn, v_grid, Uhn_addn, kx=3, ky=3, s=0.)
                idx = (S_grid - dividend_amounts[k] < 0).sum()
                X, Y = np.meshgrid(S_grid[idx:]-dividend_amounts[k], v_grid, indexing='ij')
                Uhn1[idx:, :] = spline(X, Y, grid=False)
                Uhn1[:idx, :] = Cs0_val                
                lambb = np.zeros((m1, m2+1), dtype=float) 
                Uhn1 = np.maximum(Uhn1, U0)
                Uhn = Uhn1
                k+=1
                pass
            #------------------------------------------------------------------------------
            # Finding the optimal exercise asset value for each variance level in the grid
            #------------------------------------------------------------------------------ 
            if display_exercise:
                lambb_addn = np.insert(lambb, 0, np.ones(m2+1, dtype=float), axis=0) if otype=='put'\
                             else np.insert(lambb, 0, np.zeros(m2+1, dtype=float), axis=0)
                Uhn_addn = np.insert(Uhn, 0, Cs0_val*np.ones(m2+1, dtype=float), axis=0)
                payoff_addn = np.insert(payoff, 0, Cs0_val)
                if otype == 'call' and j in (div_indices-1):
                    #---------------------------------------------------------------------------
                    # For a call option, determining the optimal exercise price of the asset
                    # through the condition V(S*, v, t_d-) = S* - K + D, where 
                    # t_d- is the calendar time just before the ex-dividend date.
                    #---------------------------------------------------------------------------
                    div_exericse_boundary = np.tile(S_grid_addn[:, np.newaxis], (1, m2+1)) - K + dividend_amounts[k-1] - Uhn_addn
                    trans_lower = np.array((div_exericse_boundary <= 0.).sum(axis=0)-1)
                elif otype=='put' and j in (div_indices-1):
                    #-------------------------------------------------------------------------------
                    # Since a put option should never be exercised before the ex-dividend date,
                    # we set the auxilary vector \lambda to be equal to 0. just before the
                    # ex-dividend date
                    #-------------------------------------------------------------------------------
                    trans_lower = - np.ones(m2+1, dtype=int)
                else:
                    if otype=='call':
                        trans_lower = np.array((lambb_addn[:-5] < math.pow(10.,-10.)).sum(axis=0)-1)
                    else:
                        trans_lower = np.where((lambb_addn[:-1] > 0) & (lambb_addn[1:] == 0))[0]
                    pass
                trans_upper = trans_lower + 1
                trans_lower = np.where(trans_lower<0, 0, trans_lower)
                trans_lower = np.where(trans_lower>m1-1, m1-1, trans_lower)
                trans_upper = np.where(trans_upper>m1-1, m1-1, trans_upper)
            
                #----------------------------------------------------------------------------------
                # Adds the current optimal exercise asset values into the exercise boundary array
                #----------------------------------------------------------------------------------   
                def find_critical_S(ll, tl, tu):
                    if tl == m1-1:
                        return S_grid_addn[-1]
                    elif tu == 0:
                        return S_grid_addn[0]
                    elif (tl==0 and tu==1) or (tl==m1-2 and tu==m1-1):
                        V_slice = Uhn_addn[[tl, tu], ll]
                        S_slice = S_grid_addn[[tl, tu]]
                        E_slice = payoff_addn[[tl, tu]]
                        D_slice = V_slice - E_slice
                        return S_slice[0] - D_slice[0]*(S_slice[1]-S_slice[0])/(D_slice[1]-D_slice[0])
                    else:
                        V_slice = Uhn_addn[tl-1: tu+2, ll]
                        S_slice = S_grid_addn[tl-1: tu+2]
                        E_slice = payoff_addn[tl-1: tu+2]
                        D_slice = V_slice - E_slice
                        cs_interp = CubicSpline(S_slice, D_slice)
                        return fsolve(cs_interp, x0=(S_slice[1]+S_slice[2])/2., maxfev=50)[0]
                if otype=='put':
                    S_exercise_t = np.array([])
                    for ll in range(m2+1):
                        S_exercise_t = np.append(S_exercise_t, find_critical_S(ll, trans_lower[ll], trans_upper[ll]))
                    pass
                else:
                    S_exercise_t = (S_grid[trans_lower]+S_grid[trans_upper])/2.
                    pass                 
                S_exercise = S_exercise_t if j==2 \
                             else np.vstack((S_exercise, S_exercise_t))  
                pass
            if otype == 'put':
                Cs0_val *= math.exp(-r*dt)
                G *= math.exp(-r*dt)  
            j+=1
            pass # <- while loop for ADI time stepping ends here
        #---------------------------------------------------------------------------
        # Interpolation function for the option price across the S and v grids
        #---------------------------------------------------------------------------
        X, Y = np.meshgrid(S_grid, v_grid, indexing='ij')
        spline_price = RectBivariateSpline(S_grid, v_grid, Uhn, kx=3, ky=3, s=0.)
        #------------------------------
        # Delta surface
        #------------------------------
        c1 = -ds[2:]/(ds[1:-1]*(ds[1:-1] + ds[2:]))
        c2 = (ds[2:] - ds[1:-1])/(ds[1:-1]*ds[2:])
        c3 = ds[1:-1]/(ds[2:]*(ds[1:-1]+ds[2:]))
        Delta = np.zeros((m1, m2+1), dtype=float)
        Delta[1:-1, :] = c1[:, None]*Uhn[:-2, :] + c2[:, None]*Uhn[1:-1, :] + c3[:, None]*Uhn[2:, :]
        Delta[0, :] = -1. if otype=='put' else 0.
        Delta[-1, :] = 1. if otype=='call' else 0.
        self.Delta = Delta
        #------------------------------
        # Vega surface
        #------------------------------
        dvol = np.sqrt(dv)
        b1 = - dvol[1:]/(dvol[:-1]*(dvol[:-1] + dvol[1:]))
        b2 = (dvol[1:] - dvol[:-1])/(dvol[1:] * dvol[:-1])
        b3 = dvol[:-1]/(dvol[1:]*(dvol[:-1] + dvol[1:]))
        Vega = np.zeros((m1, m2+1), dtype=float)
        Vega[:, 1:-1] = b1[None, :]*Uhn[:, :-2] + b2[None, :]*Uhn[:, 1:-1] + b3[None, :]*Uhn[:, 2:]
        Vega[:, 0] = (Uhn[:, 1] - Uhn[:, 0])/dvol[0]
        Vega[:, 0] = (Uhn[:, -1] - Uhn[:, -2])/dvol[-1]
        self.Vega = Vega
        self.Sv_grid = [S_grid, v_grid]
        spline_delta = RectBivariateSpline(S_grid, v_grid, Delta, kx=3, ky=3, s=0.)
        spline_vega = RectBivariateSpline(S_grid, v_grid, Vega, kx=3, ky=3, s=0.)
        #----------------------------------------------------
        # Stores the grid values into the instance variable
        # `exercise_arrays` that can be later used for 
        # plotting the optimal exercise boundary surface
        #----------------------------------------------------
        if display_exercise:
            idx = np.searchsorted(v_grid, 3.)
            v_grid = v_grid[:idx]
            S_exercise = S_exercise[:, :idx]
            t_grid = t_grid[3:]
            div_indices-=3
            self.tv_grid = [t_grid, v_grid]
            self.exercise_arrays = [div_indices, S_exercise]
            self.exercise_surface = S_exercise
        #--------------------------------------------------------------
        # Returns the interpolated option value at point (S0, v0)
        #--------------------------------------------------------------
        return spline_price([S0], [v0], grid=False)[0],\
               spline_delta([S0], [v0], grid=False)[0],\
               spline_vega([S0], [v0], grid=False)[0]

    def smoothing_exercise_surface(self):
        """Smoothing out the exercise boundary surface"""
        t_grid, v_grid = self.tv_grid
        div_indices, S_exercise = self.exercise_arrays
        otype = self.otype
        nn = len(div_indices)
        if nn>0:
            for jj in range(nn+1):
                if jj==0:
                    X, Y = np.meshgrid(t_grid[: div_indices[0]-1], v_grid, indexing='ij')
                    spline = RectBivariateSpline(t_grid[: div_indices[0]-1], v_grid,\
                                                 S_exercise[:div_indices[0]-1, :], kx=3, ky=3, s=4.0)
                    S_fitted_exercise_grid = spline(X, Y, grid=False)
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, S_exercise[div_indices[0]-1, :]))
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, S_exercise[div_indices[0], :]))
                elif jj==nn:
                    X, Y = np.meshgrid(t_grid[div_indices[-1]+1: ], v_grid, indexing='ij')
                    spline = RectBivariateSpline(t_grid[div_indices[-1]+1: ], v_grid,\
                                                 S_exercise[div_indices[-1]+1:, :], kx=3, ky=3, s=4.0)
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, spline(X, Y, grid=False)))
                else:
                    X, Y = np.meshgrid(t_grid[div_indices[jj-1]+1: div_indices[jj]-1], v_grid, indexing='ij')
                    spline = RectBivariateSpline(t_grid[div_indices[jj-1]+1: div_indices[jj]-1], v_grid, 
                                         S_exercise[div_indices[jj-1]+1: div_indices[jj]-1, :], kx=3, ky=3, s=4.0)
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, spline(X, Y, grid=False)))
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, S_exercise[div_indices[jj]-1, :]))
                    S_fitted_exercise_grid = np.vstack((S_fitted_exercise_grid, S_exercise[div_indices[jj], :]))
                pass
        else:
            X, Y = np.meshgrid(t_grid, v_grid, indexing='ij')
            spline = RectBivariateSpline(t_grid, v_grid, S_exercise, kx=3, ky=3, s=5.0)
            S_fitted_exercise_grid = spline(X, Y, grid=False)
        self.exercise_surface = S_fitted_exercise_grid
        
    def plot_exercise_boundary(self, elev=25, azim=60):
        """
        Plots the exercise boundary as a function of 
        time to maturity and variance.
        """  
        if len(self.exercise_surface)==0:
            print(f"Change the argument `dispay_exercise` of the function `price_heston`\n to `True` in order to obtain the exercise boundary.")
        t_grid, v_grid = self.tv_grid
        exercise_surface = self.exercise_surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(t_grid, v_grid, indexing='ij')
        surf = ax.plot_surface(X, Y, exercise_surface, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
        ax.set_xlabel('Time to maturity (in years)')
        ax.set_ylabel('Variance')
        ax.set_zlabel(r'Asset price (in \$)')
        ax.set_title(f"Optimal exercise boundary for {self.otype} option");
        fig.colorbar(surf)
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f"../figures/early_exercise_boundary_{self.otype}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_delta_surface(self, elev=25, azim=60):
        """
        Plots the delta surface as a function of asset
        price and volatility.
        """  
        S_grid, v_grid = self.Sv_grid
        idx = np.searchsorted(S_grid, 2.5*self.K)
        Delta = self.Delta
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(S_grid[:idx], v_grid, indexing='ij')
        surf = ax.plot_surface(X, Y, Delta[:idx, :], cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'Asset price (in \$)')
        ax.set_ylabel('Variance')
        ax.set_zlabel(r'$\Delta$')
        ax.set_title(f"Delta surface for {self.otype} option");
        fig.colorbar(surf)
        ax.view_init(elev=elev, azim=azim)
        plt.show()
    def plot_vega_surface(self, elev=25, azim=60):
        """
        Plots the vega surface as a function of asset
        price and volatility.
        """  
        S_grid, v_grid = self.Sv_grid
        Vega = self.Vega
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(S_grid, v_grid[1:-1], indexing='ij')
        surf = ax.plot_surface(X, Y, Vega[:, 1:-1], cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'Asset price (in \$)')
        ax.set_ylabel('Variance')
        ax.set_zlabel(r'Vega (in \$)')
        ax.set_title(f"Vega surface for {self.otype} option");
        fig.colorbar(surf)
        ax.view_init(elev=elev, azim=azim)
        plt.show()