import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

def magnitude_of_completeness(mags: np.ndarray, bin_width=0.1) -> float:
    """Maximum Curvature method: Mc = bin with highest frequency."""
    bins = np.arange(mags.min(), mags.max(), bin_width)
    counts, edges = np.histogram(mags, bins=bins)
    return edges[np.argmax(counts)]

def gutenberg_richter(df: pd.DataFrame, bin_width=0.1):
    mags = df["mag"].values
    Mc = magnitude_of_completeness(mags)
    mags_above = mags[mags >= Mc]

    # Cumulative frequency distribution
    bins = np.arange(Mc, mags.max() + bin_width, bin_width)
    N = np.array([np.sum(mags_above >= m) for m in bins])
    # Remove zeros before log
    mask = N > 0
    log_N = np.log10(N[mask])
    M_vals = bins[mask]

    # Linear regression
    slope, intercept, r, p, se = stats.linregress(M_vals, log_N)
    b_value = -slope
    b_uncertainty = 2.3 * b_value**2 * np.std(mags_above) / np.sqrt(len(mags_above))

    print(f"Magnitude of Completeness (Mc): {Mc:.2f}")
    print(f"b-value: {b_value:.3f} ± {b_uncertainty:.3f}")
    print(f"R²: {r**2:.4f}  (target ≥ 0.95)")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(M_vals, log_N, s=15, label="Observed", zorder=3)
    fit_line = intercept + slope * M_vals
    plt.plot(M_vals, fit_line, 'r-', lw=2,
             label=f"Fit: b={b_value:.2f}, R²={r**2:.3f}")
    plt.axvline(Mc, ls="--", color="gray", label=f"Mc={Mc:.1f}")
    plt.xlabel("Magnitude (M)")
    plt.ylabel("log₁₀ N (cumulative count)")
    plt.title("Gutenberg–Richter Relation")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/gutenberg_richter.png", dpi=150)
    plt.close()

    return b_value, b_uncertainty, r**2

if __name__ == "__main__":
    df = pd.read_csv("data/catalog.csv", parse_dates=["time"])
    gutenberg_richter(df)