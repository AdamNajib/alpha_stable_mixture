import numpy as np
from alpha_stable_mixture.em import em_stable_mixture
from alpha_stable_mixture.ecf_estimators import estimate_stable_weighted_ols


def test_em_stable_mixture_runs():
    np.random.seed(42)
    data = np.random.standard_normal(50)
    u = np.linspace(0.1, 1.5, 20)

    result = em_stable_mixture(data, u, estimator_func=estimate_stable_weighted_ols, max_iter=10)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "weights" in result and "params1" in result and "params2" in result, "Missing keys in result"
    assert 0 <= result["weights"] <= 1, "Invalid mixture weight"
    assert len(result["params1"]) == 4 and len(result["params2"]) == 4, "Invalid parameter lengths"
    print("✅ EM algorithm test passed.")


if __name__ == "__main__":
    test_em_stable_mixture_runs()
