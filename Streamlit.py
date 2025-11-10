# streamlit_app.py
# ------------------------------------------------------------
# NYC DOHMH Rodent Inspections — Interactive Dashboard (2010–2024)
# Author: <your name>
# ------------------------------------------------------------
# Features:
# - Sidebar filters: year range, borough(s), result types
# - KPIs (counts), time series by year, seasonality by month
# - Borough outcome breakdown (stacked bar)
# - Year × Month heatmap (seasonality across years)
# - Top-20 neighborhoods bar (NTA)
# - Interactive map (sampled for performance)
# - Download filtered data
#
# Data source: https://data.cityofnewyork.us/Health/DOHMH-Rodent-Inspection/p937-wjvj
# ------------------------------------------------------------

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# App config & theming
# ----------------------------
st.set_page_config(
    page_title="NYC Rodent Inspections — Dashboard",
    page_icon="🐀",
    layout="wide"
)

# Colorblind-friendly palette (Okabe–Ito / Tol blend)
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#999999"]
PAL_SEQ = px.colors.sequential.Viridis  # good for heatmaps

# ----------------------------
# Data loading
# ----------------------------
NYC_URL = "https://data.cityofnewyork.us/api/views/p937-wjvj/rows.csv?accessType=DOWNLOAD"

USECOLS = [
    "INSPECTION_TYPE","JOB_TICKET_OR_WORK_ORDER_ID","JOB_ID","JOB_PROGRESS",
    "BOROUGH","ZIP_CODE","LATITUDE","LONGITUDE","INSPECTION_DATE","RESULT","NTA"
]

DTYPES = {
    "INSPECTION_TYPE": "category",
    "JOB_TICKET_OR_WORK_ORDER_ID": "string",
    "JOB_ID": "string",
    "JOB_PROGRESS": "category",
    "BOROUGH": "category",
    "ZIP_CODE": "float64",   # keep numeric for grouping
    "LATITUDE": "float64",
    "LONGITUDE": "float64",
    "RESULT": "category",
    "NTA": "string",         # free text neighborhood
}

@st.cache_data(show_spinner=True)
def load_data(max_rows:int|None=None) -> pd.DataFrame:
    """
    Load minimal columns and parse dates.
    If a local parquet sample `rodents_sample.parquet` exists, we prefer it for speed.
    """
    # If user pre-saved a processed sample, use that (recommended for Streamlit Cloud)
    sample_path = "rodents_sample.parquet"
    if os.path.exists(sample_path):
        df = pd.read_parquet(sample_path)
    else:
        df = pd.read_csv(
            NYC_URL,
            usecols=USECOLS,
            dtype=DTYPES,
            nrows=max_rows,          # None = full dataset; set to e.g. 1_000_000 for quicker loads
            low_memory=False
        )

        # Basic cleaning
        df = df.dropna(subset=["BOROUGH","RESULT","INSPECTION_DATE"])
        df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce")
        df = df.dropna(subset=["INSPECTION_DATE"])

        # Date parts
        df["YEAR"]  = df["INSPECTION_DATE"].dt.year
        df["MONTH"] = df["INSPECTION_DATE"].dt.month

        # Normalize NTA text
        df["NTA"] = df["NTA"].fillna("Unknown").str.strip()

    # Ensure categories for compact memory
    for col in ["INSPECTION_TYPE","JOB_PROGRESS","BOROUGH","RESULT"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

with st.spinner("Loading data… (first run is cached)"):
    # Tip: If Streamlit memory is tight, set max_rows=1_000_000
    df = load_data(max_rows=None)

# Year bounds inside the dataset
MIN_YEAR, MAX_YEAR = int(df["YEAR"].min()), int(df["YEAR"].max())

# ----------------------------
# Sidebar — Filters
# ----------------------------
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Year range", min_value=MIN_YEAR, max_value=MAX_YEAR,
                               value=(max(MIN_YEAR, 2010), MAX_YEAR), step=1)

boroughs = st.sidebar.multiselect(
    "Borough",
    options=sorted(df["BOROUGH"].dropna().unique().tolist()),
    default=sorted(df["BOROUGH"].dropna().unique().tolist())
)

results = st.sidebar.multiselect(
    "Inspection Result",
    options=sorted(df["RESULT"].dropna().cat.categories.tolist()),
    default=sorted(df["RESULT"].dropna().cat.categories.tolist())
)

st.sidebar.caption("Tip: Use fewer selections to speed up plotting.")

# Apply filters
mask = (
    df["YEAR"].between(year_range[0], year_range[1]) &
    df["BOROUGH"].isin(boroughs) &
    df["RESULT"].isin(results)
)
df_f = df.loc[mask].copy()

# ----------------------------
# Header & KPIs
# ----------------------------
st.title("🐀 NYC Rodent Inspections — Interactive Dashboard")
st.caption("2010–2024 • DOHMH Open Data • All visuals are interactive (hover, zoom, legend toggle)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total inspections", f"{len(df_f):,}")
k2.metric("Boroughs selected", f"{len(boroughs)}")
k3.metric("Results selected", f"{len(results)}")
k4.metric("Year window", f"{year_range[0]}–{year_range[1]}")

st.divider()

# ----------------------------
# Time series by YEAR
# ----------------------------
ts = (df_f.groupby("YEAR").size().reset_index(name="count").sort_values("YEAR"))
fig_ts = px.line(
    ts, x="YEAR", y="count",
    markers=True,
    color_discrete_sequence=[PAL[0]],
    title="Inspections per Year"
)
fig_ts.update_layout(yaxis_title="Inspections", xaxis_title="Year", height=420)
st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------------
# Seasonality by MONTH (across selected years)
# ----------------------------
mth = (df_f.groupby("MONTH").size().reset_index(name="count").sort_values("MONTH"))
mth["MONTH_NAME"] = pd.to_datetime(mth["MONTH"], format="%m").dt.strftime("%b")
fig_m = px.line(
    mth, x="MONTH_NAME", y="count",
    markers=True,
    color_discrete_sequence=[PAL[1]],
    title="Seasonality — Inspections by Month"
)
fig_m.update_layout(xaxis_title="Month", yaxis_title="Inspections", height=420)
st.plotly_chart(fig_m, use_container_width=True)

st.divider()

# ----------------------------
# Borough × Result (stacked)
# ----------------------------
br = (
    df_f.groupby(["BOROUGH","RESULT"])
        .size().reset_index(name="count")
        .sort_values(["BOROUGH","count"], ascending=[True, False])
)
fig_stack = px.bar(
    br, x="BOROUGH", y="count", color="RESULT",
    color_discrete_sequence=PAL,
    title="Inspection Outcomes by Borough (stacked)",
    category_orders={"BOROUGH": sorted(df_f["BOROUGH"].unique().tolist())}
)
fig_stack.update_layout(yaxis_title="Inspections", xaxis_title="", legend_title="Result", height=500)
st.plotly_chart(fig_stack, use_container_width=True)

# ----------------------------
# Year × Month heatmap
# ----------------------------
ym = df_f.groupby(["YEAR","MONTH"]).size().reset_index(name="count")
ym["MONTH_NAME"] = pd.to_datetime(ym["MONTH"], format="%m").dt.strftime("%b")
fig_hm = px.density_heatmap(
    ym, x="MONTH_NAME", y="YEAR", z="count",
    color_continuous_scale=PAL_SEQ, nbinsx=12,
    title="Heatmap — Inspections by Year and Month"
)
fig_hm.update_layout(xaxis_title="Month", yaxis_title="Year", height=550)
st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ----------------------------
# Top-20 neighborhoods (NTA)
# ----------------------------
nta_ct = (
    df_f.assign(NTA=df_f["NTA"].fillna("Unknown"))
       .groupby("NTA").size().reset_index(name="count")
       .sort_values("count", ascending=False)
       .head(20).sort_values("count", ascending=True)
)
fig_nta = px.bar(
    nta_ct, x="count", y="NTA", orientation="h",
    color="count", color_continuous_scale=PAL_SEQ,
    title="Top 20 Neighborhoods by Inspection Volume"
)
fig_nta.update_layout(xaxis_title="Inspections", yaxis_title="", coloraxis_showscale=False, height=650)
st.plotly_chart(fig_nta, use_container_width=True)

# ----------------------------
# Interactive Map (sampled to keep it fast)
# ----------------------------
st.subheader("Map — Inspection Locations (sampled)")
map_n = st.slider("Max points to plot (higher = slower)", 2_000, 50_000, value=15_000, step=1_000)
df_map = (
    df_f.dropna(subset=["LATITUDE","LONGITUDE"])
        .sample(n=min(map_n, len(df_f)), random_state=42) if len(df_f) > map_n else df_f.dropna(subset=["LATITUDE","LONGITUDE"])
)

fig_map = px.scatter_mapbox(
    df_map,
    lat="LATITUDE", lon="LONGITUDE",
    color="RESULT",
    hover_data={"BOROUGH":True,"ZIP_CODE":True,"NTA":True,"INSPECTION_DATE":True},
    zoom=9, height=600,
    color_discrete_sequence=PAL
)
fig_map.update_layout(
    mapbox_style="open-street-map",
    margin=dict(l=0,r=0,t=0,b=0),
    legend_title="Result"
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ----------------------------
# Data download
# ----------------------------
st.subheader("Download filtered data")
csv = df_f[USECOLS + ["YEAR","MONTH"]].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="nyc_rodent_inspections_filtered.csv", mime="text/csv")

# ----------------------------
# About
# ----------------------------
with st.expander("About this dashboard"):
    st.markdown(
        """
**Data**: NYC DOHMH Rodent Inspection dataset (2010–2024).  
**Palette**: Color-blind-friendly.  
**Map**: OpenStreetMap (no token required).  
**Performance tips**:
- The full CSV is large. For fastest loads on Streamlit Cloud, commit a pre-filtered `rodents_sample.parquet` to your repo (e.g., ~500k rows).
- Use the sidebar to narrow years / boroughs / results.
        """
    )
