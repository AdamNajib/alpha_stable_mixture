import numpy as np
from scipy.optimize import minimize
from .utils import r_stable_pdf
from alpha_stable_mixture.utils import stable_fit_init
from scipy.stats import levy_stable

def negative_log_likelihood(params, data):
    alpha, beta, gamma, delta = params
    if not (0 < alpha <= 2 and -1 <= beta <= 1 and gamma > 0):
        return np.inf
    try:
        pdf_vals = r_stable_pdf(data, alpha, beta, gamma, delta)
        log_likelihood = np.sum(np.log(np.clip(pdf_vals, 1e-300, None)))
        return -log_likelihood
    except Exception:
        return np.inf

def fit_alpha_stable_mle(data):
    alpha0, beta0, gamma0, delta0 = stable_fit_init(data)
    bounds = [(0.1, 2.0), (-1, 1), (1e-6, None), (None, None)]
    result = minimize(
        negative_log_likelihood,
        x0=[alpha0, beta0, gamma0, delta0],
        args=(data,),
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': False, 'maxiter': 300}
    )
    return result.x.tolist() if result.success else [np.nan] * 4

def log_likelihood_mixture(params, data):
    """
    Compute the negative log-likelihood for a mixture of two stable distributions.
    """
    # Unpack 9 parameters: w + 2x (alpha, beta, scale, loc)
    w = params[0]
    a1, b1, s1, l1 = params[1:5]
    a2, b2, s2, l2 = params[5:9]

    if not (0 < w < 1 and 0.1 < a1 <= 2 and 0.1 < a2 <= 2 and -1 <= b1 <= 1 and -1 <= b2 <= 1 and s1 > 0 and s2 > 0):
        return np.inf  # invalid
    
    p1 = (a1, b1, s1, l1)
    p2 = (a2, b2, s2, l2)
    try:
        p1 = r_stable_pdf(data, *p1)
        p2 = r_stable_pdf(data, *p2)
        mix_pdf = w * p1 + (1 - w) * p2
        log_likelihood = np.sum(np.log(np.clip(mix_pdf, 1e-300, None)))
        return -log_likelihood  # for minimization
    except Exception as e:
        print("MLE error:", e)
        return np.inf


def fit_mle_mixture(data):
    init_params = [0.5, 1.3, 0.0, 1.0, -1.5, 1.7, 0.0, 1.5, 4.5]
    bounds = [
        (0.01, 0.99),         # w
        (0.1, 2.0), (-1, 1), (1e-2, None), (None, None),  # comp 1
        (0.1, 2.0), (-1, 1), (1e-2, None), (None, None),  # comp 2
    ]

    result = minimize(log_likelihood_mixture, init_params, args=(data,), bounds=bounds, method='L-BFGS-B')

    if not result.success:
        print("MLE failed:", result.message)
        return init_params
    return result.x


# Function to calculate the negative log-likelihood of the stable distribution
def L_stable(param, obs):
    """
    Computes the negative log-likelihood for a stable distribution using the R function `r_stable_pdf`.

    Parameters:
        param (list): Parameters [alpha, beta, gamma, delta].
        obs (array-like): Observed data.

    Returns:
        float: Negative log-likelihood value.
    """
    try:
        pdf_vals = r_stable_pdf(obs, *param)
        return -np.sum(np.log(np.clip(pdf_vals, 1e-300, None)))  # Avoid log(0)
    except Exception as e:
        print(f"[Error in L_stable] {e}")
        return np.inf
    
# Function to estimate the parameters using maximum likelihood
def Max_vrai(x):
    """
    Estimates the parameters of a stable distribution using maximum likelihood and the R function `r_stable_pdf`.

    Parameters:
        x (array-like): Input data.

    Returns:
        dict: Estimated parameters {alpha, beta, gamma, delta}.
    """
    init_params = stable_fit_init(x)  # Initial parameter guess
    bounds = [(0.1, 2), (-1, 1), (1e-3, None), (None, None)]  # Parameter bounds

    result = minimize(
        L_stable,
        init_params,
        args=(x,),
        method="Nelder-Mead",
        options={'maxfev': 10000, 'maxiter': 10000, 'disp': True},
        bounds=bounds
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    alpha, beta, gamma, delta = result.x
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}


def neg_log_likelihood(params, X):
    alpha, beta, gamma, delta = params
    if not (0.1 < alpha <= 2 and -1 <= beta <= 1 and gamma > 0):
        return np.inf
    return -np.sum(levy_stable.logpdf(X, alpha, beta, loc=delta, scale=gamma))

def mle_estimate(X, x0=None):
    if x0 is None:
        x0 = [1.5, 0.0, 1.0, 0.0]  # default starting guess
    result = minimize(neg_log_likelihood, x0, args=(X,), method='Nelder-Mead')
    if result.success:
        alpha, beta, gamma, delta = result.x
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
    else:
        return {"alpha": np.nan, "beta": np.nan, "gamma": np.nan, "delta": np.nan}