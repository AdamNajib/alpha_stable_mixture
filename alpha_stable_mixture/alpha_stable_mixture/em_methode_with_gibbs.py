import numpy as np
from sklearn.cluster import KMeans
from .utils import *
from .ecf_estimators import *
from .gibbs import mock_gibbs_sampling

def em_estimate_stable_recursive_ecf_with_gibbs(data, max_iter=100, tol=1e-4):
    """
    EM algorithm using recursive ECF for alpha-stable mixture with Gibbs-based M-step.
    
    Returns:
        params1, params2: dictionaries with alpha, beta, gamma, delta
        w: weight of first component
    """

    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)

    # === Initialization via clustering ===
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_recursive_ecf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_recursive_ecf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        # E-step
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        # M-step with Gibbs
        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        # Log-likelihood
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w


def em_estimate_stable_kernel_ecf_with_gibbs(data, max_iter=100, tol=1e-4):
    """
    EM algorithm with Gibbs-based M-step to fit a mixture of two alpha-stable distributions
    using the kernel-based ECF method for initialization and likelihood tracking.
    """
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)

    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_kernel_ecf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_kernel_ecf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-enhanced M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        # Log-likelihood
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_weighted_ols_with_gibbs(data, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)

    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_weighted_ols(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_weighted_ols(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-based M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_from_cdf_with_gibbs(data, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)

    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_from_cdf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_from_cdf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-based M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_stable_mixture(data, u, estimator_func=estimate_stable_weighted_ols, max_iter=300, epsilon=1e-3):
    np.random.seed(134)
    S = data
    n = len(S)

    # Initial clustering
    kmeans = KMeans(n_clusters=2, random_state=134).fit(S.reshape(-1, 1))
    labels = kmeans.labels_

    # Initial parameter estimation
    S1 = estimator_func(S[labels == 0], u)
    S2 = estimator_func(S[labels == 1], u)

    w = np.mean(labels == 0)
    p1 = [S1['alpha'], S1['beta'], ensure_positive_scale(S1['delta']), ensure_positive_scale(S1['gamma'])]
    print(f"[Cluster 1] gamma: {S1['gamma']} → adjusted: {p1[3]}")
    p2 = [S2['alpha'], S2['beta'], ensure_positive_scale(S2['delta']), ensure_positive_scale(S2['gamma'])]
    print(f"[Cluster 2] gamma: {S2['gamma']} → adjusted: {p2[3]}")

    LV = -np.inf
    for s in range(max_iter):
        cc = np.zeros(n, dtype=int)

        for i in range(n):
            try:
                v1 = np.log(w) + np.log(r_stable_pdf(S[i:i+1], *p1)[0] + 1e-10)
                v2 = np.log(1 - w) + np.log(r_stable_pdf(S[i:i+1], *p2)[0] + 1e-10)
                v = np.exp([v1, v2] - np.max([v1, v2]))
                v = v / np.sum(v) if np.sum(v) > 0 else np.array([0.5, 0.5])
                v = np.clip(v, 0, 1)
            except Exception:
                v = np.array([0.5, 0.5])

            cc[i] = np.random.choice([0, 1], p=v)

        w = np.clip(np.mean(cc == 0), 0.01, 0.99)

        if np.sum(cc == 0) >= 2:
            try:
                L1 = estimator_func(S[cc == 0], u)
                if all(np.isfinite([L1[k] for k in ['alpha', 'beta', 'delta', 'gamma']])):
                    p1 = [L1['alpha'], L1['beta'], ensure_positive_scale(L1['delta']), ensure_positive_scale(L1['gamma'])]
            except Exception:
                pass

        if np.sum(cc == 1) >= 2:
            try:
                L2 = estimator_func(S[cc == 1], u)
                if all(np.isfinite([L2[k] for k in ['alpha', 'beta', 'delta', 'gamma']])):
                    p2 = [L2['alpha'], L2['beta'], ensure_positive_scale(L2['delta']), ensure_positive_scale(L2['gamma'])]
            except Exception:
                pass

        LVn = np.sum(np.log(w * r_stable_pdf(S, *p1) + (1 - w) * r_stable_pdf(S, *p2)))
        if abs(LVn - LV) / abs(LVn) < epsilon:
            break
        LV = LVn
        print(f"Iteration {s+1}, Log-likelihood: {LVn}")

    return {
        "weights": w,
        "params1": p1,
        "params2": p2,
        "log_likelihood": LV
    }

