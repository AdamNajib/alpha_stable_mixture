import streamlit as st
import pandas as pd
import numpy as np
from alpha_stable_mixture.em import em_stable_mixture, plot_final_mixture_fit
from alpha_stable_mixture.ecf_estimators import (
    estimate_stable_kernel_ecf,
    estimate_stable_weighted_ols,
    estimate_stable_from_cdf
)
from alpha_stable_mixture.mle import fit_alpha_stable_mle
from alpha_stable_mixture.metrics import compute_model_metrics

from interface.preprocess import extract_serial_intervals

st.set_page_config(page_title="Alpha-Stable Mixture Dashboard", layout="wide")
st.title("📊 Alpha-Stable Mixture Model Dashboard")

uploaded_file = st.file_uploader("📁 Upload your data (CSV with 'x.lb', 'x.ub', 'y'):", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        data = extract_serial_intervals(df)

        st.success("✅ Data loaded and processed.")
        st.write("**Preview of Serial Intervals:**")
        st.dataframe(df[['x.lb', 'x.ub', 'y', 'SI']].head())

        u = np.linspace(0.1, 1, 20)
        methods = {
            "ECF Kernel": estimate_stable_kernel_ecf,
            "ECF Weighted OLS": estimate_stable_weighted_ols,
            "ECF from CDF": estimate_stable_from_cdf
        }

        selected_method = st.selectbox("Choose estimation method for EM:", list(methods.keys()))
        estimator_func = methods[selected_method]

        with st.spinner("🧠 Running EM algorithm..."):
            result = em_stable_mixture(data, u, estimator_func=estimator_func)

        st.success("🎯 EM estimation complete.")
        st.subheader("📋 Parameters")
        st.write("**Weights:**", result["weights"])
        st.write("**Component 1:**", result["params1"])
        st.write("**Component 2:**", result["params2"])
        st.write("**Log-likelihood:**", result["log_likelihood"])

        st.subheader("📈 Fit Plot")
        plot_final_mixture_fit(data, result["params1"], result["params2"], result["weights"])
        st.image("mixture_alpha_stable_fit_final.png")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
        st.warning("Please ensure the file contains 'x.lb', 'x.ub', 'y' in the correct format.")
else:
    st.info("Please upload a CSV file to begin.")
