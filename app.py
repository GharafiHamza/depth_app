# app.py
# One-page Streamlit app:
# - Mode 1: Preprocessed file -> validate required columns -> map immediately
# - Mode 2: Raw file -> compute Elevation, Depth, Distance to shoreline (km), Closest port (+ distance) -> map
#
# Extras:
# - Optional shoreline GeoJSON (defaults to Saudi file if present)
# - Optional ports file (defaults to saudi_ports_precise.csv if present)
# - Points colored by status: deployed=red, approved=green, else gray
# - Map persists via session_state

import io
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from shapely.geometry import shape, Point
from shapely.ops import transform as shp_transform, unary_union
import pyproj

import folium
from streamlit_folium import st_folium

# --------------------- Page/Session Setup ---------------------
st.set_page_config(page_title="Depth / Elevation / Shoreline & Ports Mapper", page_icon="🌊", layout="wide")

# Session defaults
for k, v in {
    "map_points_df": None,
    "map_ports_df": None,
    "last_processed_df": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------- Constants & Defaults -------------------
GMRT_BASE = "http://www.gmrt.org/services/PointServer"   # topo+bathy; ocean negative
OTD_BASE  = "https://api.opentopodata.org/v1/etopo1"      # fallback topo+bathy
REQ_TIMEOUT = 10
ROUND_DECIMALS_DEFAULT = 6

# Required columns for preprocessed files (case-insensitive check)
REQUIRED_PRE_COLS = [
    "elevation", "depth",
    "distance to shoreline (km)",
    "closest port", "distance from port (km)"
]

DEFAULT_SHORE_CANDIDATES = [
    Path("./Saudi_Arabia_Multipolygon_GSHHGf_NE10m.geojson"),
    Path("Saudi_Arabia_Multipolygon_GSHHGf_NE10m.geojson"),
]
DEFAULT_PORTS_CANDIDATES = [
    Path("./saudi_ports_precise.csv"),
    Path("saudi_ports_precise.csv"),
]

# Geodesic for port distances (WGS84)
GEOD = pyproj.Geod(ellps="WGS84")
# Transformer to Web Mercator for shoreline planar distances (meters) -> divide by 1000
TO_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

# --------------------- Utilities ---------------------
def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def pick_excel_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return "openpyxl"
        except Exception:
            return None

def autodetect_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None

def require_columns_case_insensitive(df: pd.DataFrame, required_lower_names: List[str]) -> Tuple[bool, List[str]]:
    have_lower = {c.lower() for c in df.columns}
    missing = [r for r in required_lower_names if r not in have_lower]
    return (len(missing) == 0, missing)

# --------------------- Caches ---------------------
cache_data = st.cache_data if hasattr(st, "cache_data") else st.cache
cache_resource = st.cache_resource if hasattr(st, "cache_resource") else st.cache

@cache_data(show_spinner=False, ttl=24 * 3600)
def gmrt_elevation(lat: float, lon: float) -> float:
    params = {"latitude": lat, "longitude": lon, "format": "geojson"}
    headers = {"Accept": "application/geo+json, application/json;q=0.9, text/plain;q=0.5"}
    r = requests.get(GMRT_BASE, params=params, headers=headers, timeout=REQ_TIMEOUT)
    try:
        js = r.json()
        if isinstance(js, dict) and js.get("features"):
            coords = js["features"][0]["geometry"]["coordinates"]
            return float(coords[2])
    except Exception:
        pass
    # Fallback: text/plain
    params["format"] = "text/plain"
    r = requests.get(GMRT_BASE, params=params, headers={"Accept": "text/plain"}, timeout=REQ_TIMEOUT)
    txt = (r.text or "").strip()
    parts = [p for p in txt.replace(",", " ").split() if p]
    if len(parts) >= 3:
        return float(parts[-1])
    raise RuntimeError(f"GMRT unexpected response: {txt[:160]}")

@cache_data(show_spinner=False, ttl=24 * 3600)
def etopo1_elevation(lat: float, lon: float) -> float:
    r = requests.get(OTD_BASE, params={"locations": f"{lat},{lon}"}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    return float(js["results"][0]["elevation"])

def get_elevation(lat: float, lon: float) -> float:
    if pd.isna(lat) or pd.isna(lon):
        return float("nan")
    try:
        return gmrt_elevation(lat, lon)
    except Exception:
        time.sleep(0.05)
        try:
            return etopo1_elevation(lat, lon)
        except Exception:
            return float("nan")

def keyify(a, b, decimals):
    try:
        return (round(float(a), decimals), round(float(b), decimals))
    except Exception:
        return (float("nan"), float("nan"))

@cache_resource(show_spinner=False)
def load_shoreline_boundary_3857(geojson_bytes: bytes):
    js = json.loads(geojson_bytes.decode("utf-8"))
    geoms = []
    if isinstance(js, dict) and js.get("type") == "FeatureCollection":
        for feat in js.get("features", []):
            if feat.get("geometry"):
                geoms.append(shape(feat["geometry"]))
    elif isinstance(js, dict) and js.get("type") == "Feature":
        geoms.append(shape(js.get("geometry")))
    else:
        geoms.append(shape(js))
    merged = unary_union(geoms)
    boundary_ll = merged.boundary
    boundary_m = shp_transform(TO_3857, boundary_ll)
    return boundary_m

def shoreline_distance_km(lon: float, lat: float, shoreline_m) -> float:
    if pd.isna(lon) or pd.isna(lat) or shoreline_m is None:
        return float("nan")
    pt_m = shp_transform(TO_3857, Point(lon, lat))
    return float(shoreline_m.distance(pt_m) / 1000.0)

def auto_build_ports_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    name_col = autodetect_col(df.columns.tolist(), ["name", "port", "port_name", "portname", "harbor", "harbour"])
    lat_col  = autodetect_col(df.columns.tolist(), ["lat", "latitude", "y"])
    lon_col  = autodetect_col(df.columns.tolist(), ["lon", "longitude", "lng", "long", "x"])
    if not (name_col and lat_col and lon_col):
        return None
    out = pd.DataFrame({
        "name": df[name_col].astype(str).str.strip(),
        "lat":  pd.to_numeric(df[lat_col], errors="coerce"),
        "lon":  pd.to_numeric(df[lon_col], errors="coerce"),
    }).dropna(subset=["lat", "lon"])
    return out

def normalize_ports_from_geojson(geojson_bytes: bytes) -> pd.DataFrame:
    js = json.loads(geojson_bytes.decode("utf-8"))
    records = []
    def pick_name(properties: dict):
        if not isinstance(properties, dict):
            return None
        for k in ["name", "Name", "NAME", "port", "PORT", "port_name", "PortName", "harbor", "harbour"]:
            if k in properties and str(properties[k]).strip():
                return str(properties[k]).strip()
        return None
    if isinstance(js, dict) and js.get("type") == "FeatureCollection":
        feats = js.get("features", [])
        for feat in feats:
            geom = feat.get("geometry")
            if not geom:
                continue
            g = shape(geom)
            if g.geom_type != "Point":
                g = g.centroid
            lon, lat = float(g.x), float(g.y)
            name = pick_name(feat.get("properties", {})) or "Unnamed port"
            records.append({"name": name, "lat": lat, "lon": lon})
    elif isinstance(js, dict) and js.get("type") == "Feature":
        geom = js.get("geometry")
        if geom:
            g = shape(geom)
            if g.geom_type != "Point":
                g = g.centroid
            lon, lat = float(g.x), float(g.y)
            name = pick_name(js.get("properties", {})) or "Unnamed port"
            records.append({"name": name, "lat": lat, "lon": lon})
    else:
        g = shape(js)
        if g.geom_type != "Point":
            g = g.centroid
        lon, lat = float(g.x), float(g.y)
        records.append({"name": "Unnamed port", "lat": lat, "lon": lon})
    return pd.DataFrame.from_records(records)

def nearest_port_for_point(lat: float, lon: float, ports_df: pd.DataFrame):
    if pd.isna(lat) or pd.isna(lon) or ports_df is None or ports_df.empty:
        return (None, float("nan"))
    plons = ports_df["lon"].to_numpy(dtype=float)
    plats = ports_df["lat"].to_numpy(dtype=float)
    _, _, dist_m = GEOD.inv(np.full(plons.shape, lon), np.full(plats.shape, lat), plons, plats)
    if dist_m.size == 0 or np.all(np.isnan(dist_m)):
        return (None, float("nan"))
    idx = int(np.nanargmin(dist_m))
    return (str(ports_df.iloc[idx]["name"]), float(dist_m[idx] / 1000.0))  # km

def color_for_status(status_value: str) -> str:
    if isinstance(status_value, str):
        s = status_value.strip().lower()
        if s == "deployed":
            return "red"
        if s == "approved":
            return "green"
    return "gray"

def _fmt_value(v, nd=4):
    if pd.isna(v):
        return "—"
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)

def popup_html_for_point_all(row: pd.Series) -> str:
    rows_html = []
    for k, v in row.items():
        rows_html.append(
            f"<tr><th style='text-align:left;padding-right:8px'>{str(k)}</th><td>{_fmt_value(v)}</td></tr>"
        )
    return f"<div style='max-height:260px;overflow:auto'><table style='font-size:12px'>{''.join(rows_html)}</table></div>"

def popup_html_for_port(row: pd.Series) -> str:
    items = [
        ("Port", row.get("name", "Unnamed")),
        ("Lat", _fmt_value(row.get("lat"))),
        ("Lon", _fmt_value(row.get("lon"))),
    ]
    rows = "".join([f"<tr><th style='text-align:left;padding-right:8px'>{k}</th><td>{v}</td></tr>" for k, v in items])
    return f"<table style='font-size:12px'>{rows}</table>"

# ---------- Base / Map builders ----------
def base_map(center=[23.8859, 45.0792], zoom=5) -> folium.Map:
    return folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

def build_folium_map(points_df: pd.DataFrame, ports_df: Optional[pd.DataFrame] = None) -> folium.Map:
    lat_col = autodetect_col(points_df.columns.tolist(), ["lat", "latitude"])
    lon_col = autodetect_col(points_df.columns.tolist(), ["lon", "longitude"])
    if not lat_col or not lon_col:
        raise ValueError("Points file must contain Lat/Lon columns (or Latitude/Longitude).")

    lats = pd.to_numeric(points_df[lat_col], errors="coerce")
    lons = pd.to_numeric(points_df[lon_col], errors="coerce")
    center = [float(lats.mean()), float(lons.mean())]
    m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

    # Points
    fg_points = folium.FeatureGroup(name="Buoys/Points", show=True)
    status_col = autodetect_col(points_df.columns.tolist(), ["status"])
    for _, row in points_df.iterrows():
        lat = row.get(lat_col); lon = row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon): continue
        color = color_for_status(row.get(status_col) if status_col else None)
        popup = folium.Popup(popup_html_for_point_all(row), max_width=360)
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5, color=color, weight=2,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=popup,
        ).add_to(fg_points)
    fg_points.add_to(m)

    # Ports (optional)
    if ports_df is not None and not ports_df.empty:
        fg_ports = folium.FeatureGroup(name="Ports", show=True)
        plat_col = autodetect_col(ports_df.columns.tolist(), ["lat", "latitude", "y"])
        plon_col = autodetect_col(ports_df.columns.tolist(), ["lon", "longitude", "x", "lng", "long"])
        pname_col = autodetect_col(ports_df.columns.tolist(), ["name", "port", "port_name", "portname"])
        for _, prow in ports_df.iterrows():
            plat = prow.get(plat_col); plon = prow.get(plon_col)
            if pd.isna(plat) or pd.isna(plon): continue
            name = prow.get(pname_col, "Port") if pname_col else "Port"
            ppopup = folium.Popup(popup_html_for_port(pd.Series({"name": name, "lat": plat, "lon": plon})), max_width=260)
            folium.Marker(
                location=[float(plat), float(plon)],
                icon=folium.Icon(color="blue", icon="info-sign"),
                popup=ppopup,
            ).add_to(fg_ports)
        fg_ports.add_to(m)

    # Legend & controls
    legend_html = """
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: white; padding: 8px 10px; border: 1px solid #888; border-radius: 6px; font-size: 12px;">
        <b>Legend</b><br>
        <span style="color:#d00;">●</span> deployed<br>
        <span style="color:#090;">●</span> approved<br>
        <span style="color:#666;">●</span> other
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    # Fit bounds
    try:
        m.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])
    except Exception:
        pass

    return m

# --------------------- UI: Top toggle ---------------------
st.subheader("Choose data source")
mode = st.radio("Data source", ["Preprocessed file", "Raw file"], horizontal=True)

# --------------------- Optional Ports & Shoreline Uploaders ---------------------
col_ps = st.columns(2)
with col_ps[0]:
    ports_up = st.file_uploader("Optional: Ports file (CSV/XLSX/GeoJSON)", type=["csv", "xlsx", "xls", "geojson", "json"])
with col_ps[1]:
    shore_up = st.file_uploader("Optional: Shoreline GeoJSON (for raw processing)", type=["geojson", "json"])

# Load optional/default Ports DF (for mapping or raw processing nearest-port computation)
ports_for_use: Optional[pd.DataFrame] = None
try:
    if ports_up is not None:
        ext = Path(ports_up.name).suffix.lower()
        if ext in [".geojson", ".json"]:
            ports_for_use = normalize_ports_from_geojson(ports_up.read())
        else:
            tmp = pd.read_excel(ports_up) if ext in [".xlsx", ".xls"] else pd.read_csv(ports_up)
            tmp_norm = auto_build_ports_df(tmp)
            ports_for_use = tmp_norm if tmp_norm is not None else tmp
    else:
        default_port_path = next((p for p in DEFAULT_PORTS_CANDIDATES if p.exists()), None)
        if default_port_path:
            tmp = pd.read_csv(default_port_path)
            tmp_norm = auto_build_ports_df(tmp)
            ports_for_use = tmp_norm if tmp_norm is not None else tmp
except Exception as e:
    st.warning(f"Could not load ports file: {e}")
# Keep a copy in session for mapping
st.session_state["map_ports_df"] = ports_for_use

# Load optional/default Shoreline (only used in raw processing)
shoreline_m = None
try:
    if shore_up is not None:
        shoreline_m = load_shoreline_boundary_3857(shore_up.read())
    else:
        default_path = next((p for p in DEFAULT_SHORE_CANDIDATES if p.exists()), None)
        if default_path:
            shoreline_m = load_shoreline_boundary_3857(default_path.read_bytes())
except Exception as e:
    st.warning(f"Could not load shoreline file: {e}")

# --------------------- MODE: PREPROCESSED ---------------------
if mode == "Preprocessed file":
    uploaded = st.file_uploader("Upload preprocessed CSV/XLSX (must include required columns)", type=["csv", "xlsx", "xls"])
    points_df = None
    if uploaded is not None:
        try:
            ext = Path(uploaded.name).suffix.lower()
            if ext == ".csv":
                points_df = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Select sheet", xls.sheet_names, index=0) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
                points_df = xls.parse(sheet_name=sheet)

            # Validate required columns (case-insensitive)
            ok1, missing_ll = require_columns_case_insensitive(points_df, ["lat", "lon", "latitude", "longitude"])
            # We accept either Lat/Lon or Latitude/Longitude, so perform custom check:
            lat_col = autodetect_col(points_df.columns.tolist(), ["lat", "latitude"])
            lon_col = autodetect_col(points_df.columns.tolist(), ["lon", "longitude"])
            have_latlon = (lat_col is not None and lon_col is not None)

            ok2, missing_proc = require_columns_case_insensitive(points_df, REQUIRED_PRE_COLS)

            if not have_latlon or not ok2:
                miss_list = []
                if not have_latlon:
                    miss_list.append("Lat/Lon (or Latitude/Longitude)")
                if not ok2:
                    miss_list.extend(missing_proc)
                st.error("Preprocessed file is missing required columns: " + ", ".join(miss_list))
            else:
                st.success(f"Preprocessed file accepted. Rows: {len(points_df)}")
                st.dataframe(points_df.head(20), use_container_width=True)
                # Put directly on the map
                st.session_state["map_points_df"] = points_df
        except Exception as e:
            st.error(f"Could not read preprocessed file: {e}")

# --------------------- MODE: RAW ---------------------
if mode == "Raw file":
    st.info("Upload raw CSV/XLSX with **Lat** and **Lon** (or **Latitude**, **Longitude**). Then click **Process & Map**.")
    uploaded = st.file_uploader("Upload raw CSV/XLSX", type=["csv", "xlsx", "xls"], key="raw_uploader")
    ROUND_DECIMALS = st.slider("Rounding for deduplication (decimal places)", 0, 8, ROUND_DECIMALS_DEFAULT)
    do_process = st.button("Process & Map")

    if uploaded is not None and do_process:
        try:
            ext = Path(uploaded.name).suffix.lower()
            if ext == ".csv":
                df_raw = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Select sheet", xls.sheet_names, index=0) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
                df_raw = xls.parse(sheet_name=sheet)
        except Exception as e:
            st.error(f"Could not read raw file: {e}")
            df_raw = None

        if df_raw is not None:
            # Validate lat/lon columns
            lat_col = autodetect_col(df_raw.columns.tolist(), ["lat", "latitude"])
            lon_col = autodetect_col(df_raw.columns.tolist(), ["lon", "longitude"])
            if not lat_col or not lon_col:
                st.error("Raw file must include **Lat** and **Lon** (or **Latitude/Longitude**).")
            else:
                lat_series = pd.to_numeric(df_raw[lat_col], errors="coerce")
                lon_series = pd.to_numeric(df_raw[lon_col], errors="coerce")

                # Elevation/Depth (dedupe queries)
                point_keys = [keyify(a, b, ROUND_DECIMALS) for a, b in zip(lat_series, lon_series)]
                unique_points = sorted({p for p in point_keys if not (pd.isna(p[0]) or pd.isna(p[1]))})

                elev_cache = {}
                prog = st.progress(0); status = st.empty()
                total = len(unique_points)
                for i, (la, lo) in enumerate(unique_points, start=1):
                    status.text(f"Elevation: querying {i}/{total}  (lat={la}, lon={lo})")
                    elev_cache[(la, lo)] = get_elevation(la, lo)
                    time.sleep(0.02)
                    prog.progress(i / max(1, total))
                status.empty(); prog.empty()

                out = df_raw.copy()
                out["Elevation"] = [
                    elev_cache.get(p, float("nan")) if not (pd.isna(p[0]) or pd.isna(p[1])) else float("nan")
                    for p in point_keys
                ]
                out["Depth"] = -out["Elevation"]

                # Distance to shoreline (km)
                if shoreline_m is not None:
                    prog2 = st.progress(0); status2 = st.empty()
                    dist_shore_km = []
                    total2 = len(out)
                    for j, (p_lon, p_lat) in enumerate(zip(lon_series, lat_series), start=1):
                        status2.text(f"Distance to shoreline: {j}/{total2}")
                        dist_shore_km.append(shoreline_distance_km(p_lon, p_lat, shoreline_m))
                        prog2.progress(j / max(1, total2))
                    status2.empty(); prog2.empty()
                else:
                    dist_shore_km = [float("nan")] * len(out)
                out["Distance to shoreline (km)"] = dist_shore_km

                # Closest port & distance (km)
                closest_names, closest_dists_km = [], []
                if ports_for_use is not None and not ports_for_use.empty:
                    pnorm = auto_build_ports_df(ports_for_use)
                    if pnorm is None:
                        pnorm = ports_for_use
                    prog3 = st.progress(0); status3 = st.empty()
                    total3 = len(out)
                    port_cache = {}
                    for k, (p_lat, p_lon) in enumerate(zip(lat_series, lon_series), start=1):
                        status3.text(f"Closest port: {k}/{total3}")
                        key = keyify(p_lat, p_lon, ROUND_DECIMALS)
                        if key in port_cache:
                            name, dist_km = port_cache[key]
                        else:
                            name, dist_km = nearest_port_for_point(p_lat, p_lon, pnorm)
                            port_cache[key] = (name, dist_km)
                        closest_names.append(name); closest_dists_km.append(dist_km)
                        prog3.progress(k / max(1, total3))
                    status3.empty(); prog3.empty()
                else:
                    closest_names = [None] * len(out)
                    closest_dists_km = [float("nan")] * len(out)

                out["Closest port"] = closest_names
                out["Distance from port (km)"] = closest_dists_km

                # Show, store, and map
                st.success(f"Processed rows: {len(out)}")
                st.dataframe(out.head(20), use_container_width=True)
                st.session_state["last_processed_df"] = out
                st.session_state["map_points_df"] = out  # map immediately

                # Downloads
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="processed_with_distances.csv", mime="text/csv")
                engine = pick_excel_engine()
                if engine:
                    xlsx_buf = io.BytesIO()
                    with pd.ExcelWriter(xlsx_buf, engine=engine) as writer:
                        out.to_excel(writer, index=False, sheet_name="Results")
                    xlsx_buf.seek(0)
                    st.download_button("⬇️ Download Excel (.xlsx)", data=xlsx_buf,
                                       file_name="processed_with_distances.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("Excel download unavailable (install: xlsxwriter or openpyxl).")

# --------------------- Map (always visible) ---------------------
st.markdown("---")
st.subheader("Map")

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    if st.button("Clear map"):
        st.session_state["map_points_df"] = None
        st.session_state["map_ports_df"] = ports_for_use  # keep ports choice
        st.experimental_rerun()

# Build and render map
points_df_for_map = st.session_state.get("map_points_df", None)
ports_df_for_map  = st.session_state.get("map_ports_df", None)

if points_df_for_map is not None and isinstance(points_df_for_map, pd.DataFrame) and len(points_df_for_map) > 0:
    try:
        m = build_folium_map(points_df_for_map, ports_df_for_map)
    except Exception as e:
        st.error(f"Failed to build map: {e}")
        m = base_map()
else:
    m = base_map()
    st.info("No data loaded yet. Upload a preprocessed file or process a raw file above to draw points.")
st_folium(m, width=None, height=700)
