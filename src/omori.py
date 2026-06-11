import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from geopy.distance import great_circle

MAINSHOCK_MAG = 5.5
RADIUS_KM     = 100
DAYS_AFTER    = 365

def omori_func(t, K, c, p):
    """Modified Omori-Utsu law."""
    return K / (c + t) ** p

def get_aftershocks(df, mainshock, radius_km=RADIUS_KM, days=DAYS_AFTER):
    ms_time = mainshock["time"]
    ms_loc  = (mainshock["latitude"], mainshock["longitude"])
    window  = df[(df["time"] > ms_time) &
                 (df["time"] < ms_time + pd.Timedelta(days=days))]
    dists = window.apply(
        lambda r: great_circle(ms_loc, (r.latitude, r.longitude)).km, axis=1
    )
    aftershocks = window[dists <= radius_km].copy()
    aftershocks["t_days"] = (aftershocks["time"] - ms_time).dt.total_seconds() / 86400
    return aftershocks

def bin_aftershocks(aftershocks, n_bins=50):
    """Return daily binned aftershock counts."""
    bins = np.linspace(0, aftershocks["t_days"].max(), n_bins)
    counts, edges = np.histogram(aftershocks["t_days"].values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    return centers[mask], counts[mask].astype(float)

def fit_omori(t, n):
    p0 = [100, 0.1, 1.0]
    bounds = ([0, 0, 0.5], [1e6, 10, 2.5])
    popt, pcov = curve_fit(omori_func, t, n, p0=p0, bounds=bounds, maxfev=5000)
    perr = np.sqrt(np.diag(pcov))
    # p-value for each parameter (H0: param=0)
    n_pts = len(t)
    dof   = n_pts - len(popt)
    t_stats = popt / perr
    p_vals  = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=dof))
    return popt, perr, p_vals

def validate_omori(df: pd.DataFrame):
    # Select well-separated mainshocks (space-time declustering: skip if
    # a larger quake occurred within 100km in prior 30 days)
    mainshocks = df[df["mag"] >= MAINSHOCK_MAG].copy()
    mainshocks = mainshocks.sort_values("mag", ascending=False).reset_index(drop=True)

    results = []
    used    = []
    fig, axes = plt.subplots(5, 1, figsize=(10, 20))

    for _, ms in mainshocks.iterrows():
        if len(results) >= 5:
            break
        # Skip if another larger event is nearby
        skip = False
        for u in used:
            dist = great_circle((ms.latitude, ms.longitude),
                                (u.latitude, u.longitude)).km
            if dist < RADIUS_KM:
                skip = True; break
        if skip:
            continue

        after = get_aftershocks(df, ms)
        if len(after) < 30:          # need enough aftershocks to fit
            continue

        t, n = bin_aftershocks(after)
        try:
            popt, perr, p_vals = fit_omori(t, n)
        except RuntimeError:
            continue

        K, c, p = popt
        results.append({"event": ms["id"], "mag": ms["mag"],
                         "K": K, "c": c, "p": p,
                         "p_K": p_vals[0], "p_c": p_vals[1], "p_p": p_vals[2]})
        used.append(ms)

        # Plot
        ax = axes[len(results)-1]
        ax.scatter(t, n, s=10, alpha=0.6, label="Binned counts")
        t_fit = np.linspace(t.min(), t.max(), 300)
        ax.plot(t_fit, omori_func(t_fit, *popt), 'r-', lw=2,
                label=f"Omori fit: K={K:.0f}, c={c:.2f}, p={p:.2f}")
        ax.set_title(f"M{ms['mag']} mainshock — {ms.get('place','')}")
        ax.set_xlabel("Days after mainshock")
        ax.set_ylabel("Aftershock rate")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/omori_fits.png", dpi=150)
    plt.close()

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    result_df.to_csv("outputs/omori_results.csv", index=False)
    return result_df

if __name__ == "__main__":
    df = pd.read_csv("data/catalog.csv", parse_dates=["time"])
    validate_omori(df)