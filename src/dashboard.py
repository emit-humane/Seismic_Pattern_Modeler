import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

def color_by_magnitude(mag: float) -> str:
    if mag < 2.5: return "#2ecc71"   # green
    elif mag < 3.5: return "#f1c40f" # yellow
    elif mag < 4.5: return "#e67e22" # orange
    elif mag < 5.5: return "#e74c3c" # red
    else: return "#8e44ad"           # purple

def build_dashboard(df: pd.DataFrame, out_path="outputs/seismic_map.html"):
    # ---- Base map centred on California ----
    m = folium.Map(location=[37.0, -119.5], zoom_start=6,
                   tiles="CartoDB dark_matter")

    # ---- 1. Heat map layer (density) ----
    heat_data = df[["latitude","longitude","mag"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=10,
            gradient={0.2:"blue",0.5:"lime",0.8:"orange",1.0:"red"},
            name="Earthquake Density").add_to(m)

    # ---- 2. M≥4 events as circle markers ----
    fg_medium = folium.FeatureGroup(name="M≥4 Events", show=True)
    for _, row in df[df["mag"] >= 4.0].iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=max(2, (row.mag - 3) * 3),
            color=color_by_magnitude(row.mag),
            fill=True, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>M{row.mag:.1f}</b><br>"
                f"{row.get('place','')}<br>"
                f"{str(row.time)[:10]}<br>"
                f"Depth: {row.depth:.1f} km",
                max_width=200),
        ).add_to(fg_medium)
    fg_medium.add_to(m)

    # ---- 3. Major events M≥5.5 with star icons ----
    fg_major = folium.FeatureGroup(name="M≥5.5 Major Quakes", show=True)
    for _, row in df[df["mag"] >= 5.5].iterrows():
        folium.Marker(
            location=[row.latitude, row.longitude],
            icon=folium.Icon(icon="star", color="red", prefix="fa"),
            popup=folium.Popup(
                f"<b>⚠ M{row.mag:.1f}</b><br>"
                f"{row.get('place','')}<br>{str(row.time)[:10]}",
                max_width=250),
        ).add_to(fg_major)
    fg_major.add_to(m)

    # ---- 4. Colormap legend ----
    colormap = cm.LinearColormap(
        colors=["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"],
        vmin=1.5, vmax=7.0,
        caption="Earthquake Magnitude")
    colormap.add_to(m)

    # ---- 5. Layer control ----
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_path)
    print(f"Dashboard saved → {out_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/catalog.csv", parse_dates=["time"])
    build_dashboard(df)