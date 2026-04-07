"""
Simple timing comparison for option pricing methods.

Measures execution time for different pricing methods.
"""

import time
import numpy as np
import pandas as pd
import optionfft as opt

# Basic parameters
initial_price, interest, vol, time_horizon = 100, 0.05, 0.25, 1
vg_nu, vg_theta = 2, -0.1

# Models
gbm_model = opt.GeometricBrownianMotion(initial_price, interest, vol)
vg_model = opt.VarianceGamma(initial_price, interest, vol, vg_theta, vg_nu)

# Option objects
gbm_call = opt.EuCall(100, time_horizon, gbm_model)
vg_call = opt.EuCall(100, time_horizon, vg_model)

# Methods to time
gbm_methods = [
    "black_scholes_price",
    "monte_carlo_price",
    "cdfFTPrice",
    "CMFTPrice"
]

vg_methods = [
    "VG_analytic_price",
    "monte_carlo_price",
    "cdfFTPrice",
    "CMFTPrice"
]

# Store timing results
gbm_times = []
vg_times = []

# Timing function
def measure_time(method_name, option_obj, runs=5):
    start = time.time()
    for _ in range(runs):
        getattr(opt.EuCall, method_name)(option_obj)
    end = time.time()
    return (end - start) / runs

# Measure GBM
for method in gbm_methods:
    t = measure_time(method, gbm_call)
    gbm_times.append(t)

# Measure VG
for method in vg_methods:
    t = measure_time(method, vg_call)
    vg_times.append(t)

# Add FFT timing
start = time.time()
opt.FFTPrice(gbm_model, time_horizon, 50, 150)
gbm_times.append(time.time() - start)

start = time.time()
opt.FFTPrice(vg_model, time_horizon, 50, 150)
vg_times.append(time.time() - start)

# Labels
labels = ["BS/Analytic", "Monte Carlo", "Fourier", "Modified", "FFT"]

# Create dataframe
timing_df = pd.DataFrame({
    "Method": labels,
    "GBM Time (s)": gbm_times,
    "VG Time (s)": vg_times
})

# Save results
timing_df.to_csv("Analysis/timing_results.csv", index=False)
