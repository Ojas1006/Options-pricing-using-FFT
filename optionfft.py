#!/usr/bin/env python
"""Implementation of Fast Fourier Transform method for European call options.

This module provides:
    1. Classes for the Geometric Brownian Motion (GBM) and Variance Gamma (VG) processes
    2. European call option pricing via Monte Carlo, Black-Scholes (GBM only),
       Fourier Inversion (cdfFT), Modified Call FT (CMFT), VG Analytic, and FFT methods
    3. Utility functions for the FFT strike grid and batch pricing

Bug fixes vs original repo
---------------------------
* black_scholes_price: operator precedence bug in d2 formula fixed
  - WRONG (original): d2 = ... / sigma * np.sqrt(T)   (gives a*sqrt(T)/sigma)
  - CORRECT:          d2 = ... / (sigma * np.sqrt(T))  (gives a/(sigma*sqrt(T)))
  The bug was silent for T=1 because sqrt(1)=1; it matters for all other maturities.
* raise statements changed from bare strings to proper RuntimeError exceptions.

Reference
---------
Carr, P. and Madan, D.B. (1999). Option Valuation Using the Fast Fourier Transform.
Journal of Computational Finance, 2(4), 61-73.
http://faculty.baruch.cuny.edu/lwu/890/CarrMadan99.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.integrate import quad
from scipy.stats import gamma, norm


# ---------------------------------------------------------------------------
# Stock Price Process Classes
# ---------------------------------------------------------------------------

class GeometricBrownianMotion:
    """Geometric Brownian Motion -- the standard BSM stock price process.

    Parameters / Attributes
    -----------------------
    S0    : float  Initial stock price.
    r     : float  Continuously compounded risk-free interest rate.
    sigma : float  Constant volatility parameter.
    """

    def __init__(self, S0, r, sigma):
        self.S0    = S0
        self.r     = r
        self.sigma = sigma

    def phi(self, t, u):
        """Characteristic function of log(S_t) under GBM.

        For S_t = S0 * exp((r - sigma^2/2)*t + sigma*W_t), the log price is
        Normal(mu, sigma^2*t) where mu = log(S0) + (r - sigma^2/2)*t.

        Parameters
        ----------
        t : float or array-like  -- Time horizon.
        u : complex or array-like -- Frequency variable.

        Returns
        -------
        complex or array-like -- phi_t(u) = E[exp(iu*log(S_t))]
        """
        S0, r, sigma = self.S0, self.r, self.sigma
        mu  = np.log(S0) + (r - 0.5 * sigma**2) * t
        var = t * sigma**2
        return np.exp(1j * u * mu - 0.5 * u**2 * var)

    def sample_path(self, T, N=200, plot=False):
        """Simulate a GBM sample path and return the terminal stock price.

        Parameters
        ----------
        T    : float  -- Terminal time.
        N    : int    -- Number of time steps (default 200).
        plot : bool   -- If True, display the simulated path.

        Returns
        -------
        float -- Simulated terminal stock price S_T.
        """
        S0, r, sigma = self.S0, self.r, self.sigma
        dt = T / N
        t  = np.linspace(0, T, N + 1)
        dW = np.random.normal(0, np.sqrt(dt), N)
        W  = np.insert(np.cumsum(dW), 0, 0)
        S  = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        if plot:
            plt.plot(t, S)
            plt.xlabel("Time"); plt.ylabel("Stock Price")
            plt.title("GBM Sample Path"); plt.show()
        return S[-1]


class VarianceGamma:
    """Variance-Gamma (VG) stock price process.

    The stock price is modelled as:
        S_t = S0 * exp((r + omega)*t + X_t(sigma, theta, nu))

    where X_t is Arithmetic Brownian motion with drift theta and volatility sigma
    time-changed by a Gamma process with mean rate 1 and variance rate nu.
    omega is the martingale correction: omega = (1/nu)*log(1 - theta*nu - 0.5*nu*sigma^2).

    Parameters / Attributes
    -----------------------
    S0    : float  Initial stock price.
    r     : float  Risk-free rate.
    sigma : float  Volatility of Brownian component (controls overall vol).
    theta : float  Drift of Brownian component (controls skewness).
    nu    : float  Variance rate of Gamma subordinator (controls kurtosis).
    omega : float, optional  Martingale correction (auto-computed if None).
    """

    def __init__(self, S0, r, sigma, theta, nu, omega=None):
        self.S0    = S0
        self.r     = r
        self.sigma = sigma
        self.theta = theta
        self.nu    = nu
        if omega is None:
            omega = (1 / nu) * np.log(1 - theta * nu - 0.5 * nu * sigma**2)
        self.omega = omega

    def phi(self, t, u):
        """Characteristic function of log(S_t) under the VG model.

        phi_T(u) = exp(iu*(log(S0) + (r+omega)*T)) / (1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)^(T/nu)

        Parameters
        ----------
        t : float or array-like
        u : complex or array-like

        Returns
        -------
        complex or array-like
        """
        S0, theta, sigma, nu = self.S0, self.theta, self.sigma, self.nu
        r, omega = self.r, self.omega
        denom = np.power(
            (1 - 1j * theta * nu * u + 0.5 * (u * sigma)**2 * nu),
            t / nu
        )
        return np.exp(1j * u * (np.log(S0) + (r + omega) * t)) / denom

    def sample_path(self, T, N=200, plot=False):
        """Simulate a VG sample path and return the terminal stock price.

        Parameters
        ----------
        T    : float  -- Terminal time.
        N    : int    -- Number of time steps.
        plot : bool   -- Plot if True.

        Returns
        -------
        float -- Simulated terminal stock price S_T.
        """
        S0, theta, sigma, nu = self.S0, self.theta, self.sigma, self.nu
        r, omega = self.r, self.omega
        dt = T / N
        t  = np.linspace(0, T, N + 1)
        Z  = np.random.normal(0, 1, N)
        dG = np.random.gamma(dt / nu, nu, N)      # Gamma time increments
        X  = theta * np.cumsum(dG) + sigma * np.cumsum(np.sqrt(dG) * Z)
        X  = np.insert(X, 0, 0)
        S  = S0 * np.exp((r + omega) * t + X)
        if plot:
            plt.plot(t, S)
            plt.xlabel("Time"); plt.ylabel("Stock Price")
            plt.title("VG Sample Path"); plt.show()
        return S[-1]


# ---------------------------------------------------------------------------
# European Call Option Class
# ---------------------------------------------------------------------------

class EuCall:
    """European Call option, supporting multiple pricing methods.

    Parameters / Attributes
    -----------------------
    K : float  Strike price.
    T : float  Time to maturity (in years).
    S : GeometricBrownianMotion or VarianceGamma  Underlying process.
    """

    def __init__(self, K, T, S):
        self.K = K
        self.T = T
        self.S = S

    def payoff(self, ST):
        """Terminal payoff: max(S_T - K, 0)."""
        return max(ST - self.K, 0)

    def monte_carlo_price(self, n=1000):
        """Price via Monte Carlo simulation.

        Parameters
        ----------
        n : int  Number of simulated paths (default 1000).

        Returns
        -------
        float  Estimated call price.
        """
        r, T = self.S.r, self.T
        payoffs = np.array([self.payoff(self.S.sample_path(T)) for _ in range(n)])
        return np.exp(-r * T) * payoffs.mean()

    def black_scholes_price(self):
        """Analytical Black-Scholes price (GBM underlying only).

        C = S0*N(d1) - K*e^(-rT)*N(d2)
        d1 = [log(S0/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        BUG FIX (vs original):
            Original code: d2 = ... / sigma * np.sqrt(T)
            This evaluates as (... / sigma) * sqrt(T) = a*sqrt(T)/sigma,
            which equals the correct answer ONLY when T=1.
            Correct code:  d2 = ... / (sigma * np.sqrt(T))

        Returns
        -------
        float  Black-Scholes call price.
        """
        if not isinstance(self.S, GeometricBrownianMotion):
            raise RuntimeError("black_scholes_price requires a GBM underlying.")
        K, T, S = self.K, self.T, self.S
        S0, r, sigma = S.S0, S.r, S.sigma
        # FIXED: parentheses added around (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d1 = d2 + sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def cdfFTPrice(self, lower=1e-8, upper=np.inf, weight=None, wvar=None):
        """Price via Fourier inversion of the delta and in-the-money probability.

        C = S0*delta - K*e^(-rT)*PrITM

        where delta and PrITM are each computed as:
            0.5 + (1/pi) * integral of a real-valued integrand.

        Note: the integrand has a singularity at u=0 (1/u term), so FFT
        cannot be applied directly. Use lower=1e-8 to avoid division by zero.

        Parameters
        ----------
        lower  : float  Lower integration bound (default 1e-8).
        upper  : float  Upper integration bound.
        weight : str    Optional scipy weight ('cauchy' can help near u=0).
        wvar   : float  Variable for the weight function.

        Returns
        -------
        float  Call price.
        """
        S0, r, K, T = self.S.S0, self.S.r, self.K, self.T
        k   = np.log(K)
        phi = self.S.phi

        def PrITMIntegrand(u):
            return np.real(-1j * np.exp(-1j * u * k) * phi(T, u) / u)

        def deltaIntegrand(u):
            num = -1j * np.exp(-1j * u * k) * phi(T, u - 1j)
            return np.real(num / (u * phi(T, -1j)))

        if weight == "cauchy":
            wvar = 0
        intITM   = quad(PrITMIntegrand, a=lower, b=upper, weight=weight, wvar=wvar)[0]
        intDelta = quad(deltaIntegrand,  a=lower, b=upper, weight=weight, wvar=wvar)[0]
        return S0 * (0.5 + intDelta / np.pi) - K * np.exp(-r * T) * (0.5 + intITM / np.pi)

    def MCallFT(self, v, alpha):
        """Fourier transform psi_T(v) of the modified call c_T(k) = e^(alpha*k)*C_T(k).

        psi_T(v) = e^(-rT) * phi_T(v - (alpha+1)*i) / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)

        Parameters
        ----------
        v     : float or array-like  Frequency variable.
        alpha : float                Damping coefficient (must be > 0).

        Returns
        -------
        complex or array-like  psi_T(v).
        """
        T, r = self.T, self.S.r
        denom = (alpha**2 + alpha - v**2) + (2 * alpha + 1) * v * 1j
        return np.exp(-r * T) * self.S.phi(T, v - (alpha + 1) * 1j) / denom

    def CMFTPrice(self, alpha=1.5):
        """Price via quadrature inversion of the modified call FT (no FFT).

        C_T(k) = (e^(-alpha*k) / pi) * integral_0^inf Re[e^(-ivk) * psi_T(v)] dv

        This is the accuracy benchmark for the FFT method (same formula,
        but using scipy quad instead of the FFT summation approximation).

        Parameters
        ----------
        alpha : float  Damping coefficient (default 1.5).

        Returns
        -------
        float  Call price.
        """
        k = np.log(self.K)

        def integrand(v):
            return np.real(np.exp(-1j * v * k) * self.MCallFT(v, alpha))

        return np.exp(-alpha * k) * quad(integrand, 0, np.inf)[0] / np.pi

    def VG_analytic_price(self):
        """VG call price via the Psi-function formula (Madan, Carr & Chang 1998).

        Uses a 1-D numerical integral rather than the hypergeometric closed form
        (which has a singularity at u=1 and is impractical; see Matsuda 2004).

        Returns
        -------
        float  VG call price.
        """
        if not isinstance(self.S, VarianceGamma):
            raise RuntimeError("VG_analytic_price requires a VG underlying.")
        K, T, S = self.K, self.T, self.S
        theta, sigma, nu = S.theta, S.sigma, S.nu
        r, S0 = S.r, S.S0

        zeta  = -theta / sigma**2
        s     = sigma / np.sqrt(1 + 0.5 * theta**2 * nu / sigma**2)
        alpha = zeta * s
        c1    = 0.5 * nu * (alpha + s)**2
        c2    = 0.5 * nu * alpha**2
        log_c = 1 + nu * (theta - 0.5 * sigma**2)
        d     = (1 / s) * (np.log(S0 / K) + r * T + (T / nu) * np.log(log_c))

        def Psi(a, b, c):
            def integrand(u):
                return norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * gamma.pdf(u, a=c)
            return quad(integrand, 0, np.inf)[0]

        c_shape = T / nu
        delta  = Psi(d * np.sqrt((1 - c1) / nu), (alpha + s) * np.sqrt(nu / (1 - c1)), c_shape)
        PrITM  = Psi(d * np.sqrt((1 - c2) / nu),  alpha      * np.sqrt(nu / (1 - c2)), c_shape)
        return S0 * delta - K * np.exp(-r * T) * PrITM


# ---------------------------------------------------------------------------
# Module-level FFT functions
# ---------------------------------------------------------------------------

def MCallFTo(S, T, v, alpha):
    """Module-level Fourier transform of the modified call (for FFTPrice).

    Parameters
    ----------
    S     : GBM or VG instance.
    T     : float   Maturity.
    v     : array   Frequency grid.
    alpha : float   Damping coefficient.

    Returns
    -------
    array  psi_T(v) values.
    """
    denom = (alpha**2 + alpha - v**2) + (2 * alpha + 1) * v * 1j
    return np.exp(-S.r * T) * S.phi(T, v - (alpha + 1) * 1j) / denom


def logStrikePartition(eta=0.25, N=4096):
    """Build the log-strike grid required by FFTPrice.

    The FFT constraint  eta * lambda = 2*pi/N  links the integration spacing
    eta to the log-strike spacing lambda.  Log-strikes run on [-b, b) where
    b = pi/eta.

    Parameters
    ----------
    eta : float  Integration spacing (default 0.25 from Carr-Madan 1999).
    N   : int    FFT size; must be a power of 2 (default 4096).

    Returns
    -------
    b    : float      Half-range of log-strike grid.
    lamb : float      Log-strike spacing lambda.
    k    : np.ndarray N log-strike values uniformly spaced in [-b, b).
    """
    b    = np.pi / eta
    lamb = 2 * np.pi / (eta * N)
    k    = -b + lamb * np.arange(N)
    return b, lamb, k


def FFTPrice(S, T, L=0, U=np.inf, alpha=1.5, eta=0.25, N=4096):
    """Price European calls across a strike range using the Carr-Madan FFT method.

    Implements equation (24) of Carr & Madan 1999 with Simpson's rule weights:

        C(k_u) = (e^(-alpha*k_u) / pi) * FFT[ e^(i*b*v_j) * psi(v_j) * w_j ]_u

    where w_j = (eta/3) * (3 + (-1)^j) with the first weight reduced by eta/3
    for the Kronecker delta term (Simpson's rule applied to the trapezoidal sum).

    Parameters
    ----------
    S     : GBM or VG instance.
    T     : float  Maturity.
    L, U  : float  Return prices only for strikes in the open interval (L, U).
    alpha : float  Damping coefficient (default 1.5).
    eta   : float  Integration spacing (default 0.25).
    N     : int    FFT size; power of 2 (default 4096).

    Returns
    -------
    np.ndarray  Call prices for strikes strictly between L and U.
    """
    # Integration frequency grid: v_j = eta * j
    V = np.arange(N) * eta

    # Log-strike grid
    b, _lamb, k = logStrikePartition(eta, N)

    # Simpson's rule weights: w_j = (eta/3) * (3 + (-1)^j) - (eta/3)*delta_{j,0}
    alternating      = np.ones(N)
    alternating[::2] = -1                      # even indices get -1, odd get +1
    weights          = (eta / 3) * (3 + alternating)
    weights[0]      -= eta / 3                 # Kronecker delta correction at j=0

    # FFT computation
    x           = np.exp(1j * b * V) * MCallFTo(S, T, V, alpha) * weights
    call_prices = np.real(np.exp(-alpha * k) / np.pi * fft(x))

    # Return only prices in the specified strike range
    mask = np.logical_and(np.exp(k) > L, np.exp(k) < U)
    return call_prices[mask]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    S0, r, K, T = 100, 0.05, 100, 1.0

    # GBM
    gbm      = GeometricBrownianMotion(S0, r, 0.2)
    call_gbm = EuCall(K, T, gbm)
    bs       = call_gbm.black_scholes_price()
    fft_g    = FFTPrice(gbm, T, L=95, U=105)
    mid      = len(fft_g) // 2
    print("=== GBM (sigma=0.20) ===")
    print(f"  Black-Scholes     : {bs:.4f}")
    print(f"  FFT (nearest ATM) : {fft_g[mid]:.4f}")

    # VG
    vg      = VarianceGamma(S0, r, 0.25, -0.1, 2.0)
    call_vg = EuCall(K, T, vg)
    print("\n=== VG (sigma=0.25, nu=2.0, theta=-0.10) ===")
    print(f"  Analytic    : {call_vg.VG_analytic_price():.4f}")
    print(f"  CMFT        : {call_vg.CMFTPrice():.4f}")
    print(f"  Monte Carlo : {call_vg.monte_carlo_price(n=3000):.4f}")
