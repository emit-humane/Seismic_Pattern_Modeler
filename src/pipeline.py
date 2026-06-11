import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
RAW_PATH = Path("data/raw_catalog.csv")

def fetch_year(year: int) -> pd.DataFrame:
    """Fetch one calendar year, M≥1.5, California region."""
    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime":   f"{year}-12-31",
        "minlatitude": 32, "maxlatitude": 42,
        "minlongitude": -125, "maxlongitude": -114,
        "minmagnitude": 1.5,
        "orderby": "time-asc",
        "limit": 20000,
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def fetch_catalog(start=1994, end=2024) -> pd.DataFrame:
    frames = []
    for year in tqdm(range(start, end + 1), desc="Fetching years"):
        try:
            frames.append(fetch_year(year))
        except Exception as e:
            print(f"  Year {year} failed: {e}")
    return pd.concat(frames, ignore_index=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.dropna(subset=["latitude", "longitude", "mag", "depth"])
    df = df[df["mag"] >= 1.5]                   # enforce floor
    df = df.drop_duplicates(subset=["time","latitude","longitude"])
    df = df.sort_values("time").reset_index(drop=True)
    # keep only earthquake-type events
    if "type" in df.columns:
        df = df[df["type"] == "earthquake"]
    return df[["time","latitude","longitude","depth","mag","place","id"]]

if __name__ == "__main__":
    if not RAW_PATH.exists():
        raw = fetch_catalog()
        raw.to_csv(RAW_PATH, index=False)
    df = clean(pd.read_csv(RAW_PATH))
    df.to_csv("data/catalog.csv", index=False)
    print(f"Catalog ready: {len(df):,} events")