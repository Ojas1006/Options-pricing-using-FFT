"""
Simple error comparison for option pricing methods.

Computes average absolute and relative error and saves results.
"""

import numpy as np
import pandas as pd
import optionfft as opt

# Basic parameters
initial_price, interest, vol, time = 100, 0.05, 0.1, 1
vol, vg_nu, vg_theta = 0.25, 2, -0.1

# Create models
gbm_model = opt.GeometricBrownianMotion(initial_price, interest, vol)
vg_model = opt.VarianceGamma(initial_price, interest, vol, vg_theta, vg_nu)

# Create call option objects (strike will change later)
gbm_call = opt.EuCall(0, time, gbm_model)
vg_call = opt.EuCall(0, time, vg_model)

# FFT settings
damping = 1.5
step = 0.25
points = 4096

# Strike range limits
low_mult = 0.5
high_mult = 2
low_strike, high_strike = low_mult * initial_price, high_mult * initial_price

# Generate strike values
log_strikes = opt.logStrikePartition(step, points)[2]
valid_idx = np.logical_and(np.exp(log_strikes) > low_strike,
                           np.exp(log_strikes) < high_strike)
strike_vals = np.exp(log_strikes)[valid_idx]

# FFT prices
fft_gbm_vals = opt.FFTPrice(gbm_model, time, low_strike, high_strike)
fft_vg_vals = opt.FFTPrice(vg_model, time, low_strike, high_strike)

# Methods to compare
gbm_methods_list = [
    "black_scholes_price",
    "monte_carlo_price",
    "cdfFTPrice",
    "CMFTPrice"
]

vg_methods_list = [
    "VG_analytic_price",
    "monte_carlo_price",
    "cdfFTPrice",
    "CMFTPrice"
]

# Prepare storage
gbm_cols = len(gbm_methods_list) + 1
vg_cols = len(vg_methods_list) + 1
num_strikes = strike_vals.shape[0]

gbm_results = np.zeros((num_strikes, gbm_cols))
vg_results = np.zeros((num_strikes, vg_cols))

# Compute prices
for idx, strike_val in enumerate(strike_vals):
    gbm_call.K = strike_val
    vg_call.K = strike_val

    # GBM methods
    for m_idx, method_name in enumerate(gbm_methods_list):
        gbm_results[idx, m_idx] = getattr(opt.EuCall, method_name)(gbm_call)
    gbm_results[idx, -1] = fft_gbm_vals[idx]

    # VG methods
    for m_idx, method_name in enumerate(vg_methods_list):
        vg_results[idx, m_idx] = getattr(opt.EuCall, method_name)(vg_call)
    vg_results[idx, -1] = fft_vg_vals[idx]

# Column names
gbm_labels = [
    "Black-Scholes",
    "Monte Carlo",
    "Fourier",
    "Modified",
    "FFT"
]

vg_labels = [
    "Analytic",
    "Monte Carlo",
    "Fourier",
    "Modified",
    "FFT"
]

# Save all prices
combined_data = np.hstack((strike_vals.reshape(num_strikes, 1),
                          gbm_results, vg_results))

column_names = ["Strike"] +                ["GBM " + name for name in gbm_labels] +                ["VG " + name for name in vg_labels]

df_all = pd.DataFrame(data=combined_data, columns=column_names)
df_all.to_csv("Analysis/all_prices.csv", index=False)

# Separate dataframes
df_gbm = pd.DataFrame(data=gbm_results, columns=gbm_labels)
df_vg = pd.DataFrame(data=vg_results, columns=vg_labels)

# GBM error
gbm_abs_err = abs(df_gbm["Black-Scholes"] - df_gbm["FFT"])
gbm_rel_err = gbm_abs_err / df_gbm["Black-Scholes"]

# VG theoretical handling
threshold = 0.03
mc_vs_analytic = np.abs((df_vg["Monte Carlo"] - df_vg["Analytic"]) /
                        df_vg["Monte Carlo"])

alt_prices = (df_vg["Fourier"] + df_vg["Modified"]) / 2

vg_true_price = np.where(
    mc_vs_analytic > threshold,
    alt_prices,
    df_vg["Analytic"]
)

df_vg["True"] = vg_true_price

# VG error
vg_abs_err = abs(vg_true_price - df_vg["FFT"])
vg_rel_err = vg_abs_err / vg_true_price

# Final error table
error_values = np.array([
    [gbm_abs_err.mean(), gbm_rel_err.mean()],
    [vg_abs_err.mean(), vg_rel_err.mean()]
])

error_df = pd.DataFrame(
    error_values,
    index=["GBM", "VG"],
    columns=["Absolute", "Relative"]
)

# Format numbers
error_df["Absolute"] = error_df["Absolute"].apply("{:.4e}".format)
error_df["Relative"] = error_df["Relative"].apply("{:.2%}".format)

# Convert to LaTeX
table_caption = "Error comparison over {:d} strikes between {:.2f} and {:.2f}".format(
    num_strikes, strike_vals[0], strike_vals[-1]
)

latex_output = error_df.to_latex(
    caption=table_caption,
    label="tab:error"
)

# Replace formatting
latex_output = latex_output.replace(r"\toprule", r"\hline\n\hline")
latex_output = latex_output.replace(r"\midrule", r"\hline")
latex_output = latex_output.replace(r"\bottomrule", r"\hline")
latex_output = latex_output.replace("Relative", r"\textbf{Relative}")
latex_output = latex_output.replace("Absolute", r"\textbf{Absolute}")
latex_output = latex_output.replace(r"\begin{table}", r"\begin{table}[h]")

# Save LaTeX file
with open("Analysis/error_table.tex", "w+") as file:
    file.write(latex_output)
