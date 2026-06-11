import sys
from pathlib import Path
import pandas as pd

# Windows consoles default to cp1252, which can't encode the Unicode symbols
# (≥, ², ±) used in the status output. Force UTF-8 for all stdout/stderr.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

Path("data").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# -- Step 1: Pipeline --
from src.pipeline import fetch_catalog, clean
if not Path("data/catalog.csv").exists():
    raw = fetch_catalog(1994, 2024)
    raw.to_csv("data/raw_catalog.csv", index=False)
df = clean(pd.read_csv("data/raw_catalog.csv"))
df.to_csv("data/catalog.csv", index=False)
print(f"[1] Catalog: {len(df):,} events\n")

# -- Step 2: Gutenberg-Richter --
from src.gutenberg_richter import gutenberg_richter
b, b_err, r2 = gutenberg_richter(df)
print(f"[2] G-R: b={b:.3f}±{b_err:.3f}, R²={r2:.3f}\n")
assert r2 >= 0.95, f"R² target not met: {r2:.3f}"

# -- Step 3: Omori --
from src.omori import validate_omori
omori_df = validate_omori(df)
print(f"[3] Omori sequences validated: {len(omori_df)}\n")
assert len(omori_df) >= 5, "Need ≥5 Omori sequences"

# -- Step 4: Model --
from src.model import train_model
model, rmse = train_model(df)
print(f"[4] XGBoost RMSE: {rmse:.2f}\n")

# -- Step 5: Dashboard --
from src.dashboard import build_dashboard
build_dashboard(df)
print("[5] Dashboard ready → outputs/seismic_map.html\n")

print("All steps complete.")