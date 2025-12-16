import math
import time

import pandas as pd
import streamlit as st
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import googlemaps

st.set_page_config(page_title="FATECs por raio + Google Distance Matrix", layout="wide")


# ---------- Utilitários ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return df


def build_map(df_in, center_lat, center_lon, radius_km, col_name_dist="dist_km"):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    folium.Marker(
        [center_lat, center_lon],
        popup=f"Centro ({center_lat:.5f}, {center_lon:.5f})",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Circle(
        [center_lat, center_lon],
        radius=radius_km * 1000,
        color="red",
        fill=True,
        fill_opacity=0.15,
    ).add_to(m)

    # Marcadores (verde dentro, azul fora)
    for _, row in df_in.iterrows():
        color = "green" if row["dentro_raio"] else "blue"
        dist_txt = f"{row[col_name_dist]:.1f} km" if pd.notna(row[col_name_dist]) else "?"
        popup = f"{row.get('unidade','(sem nome)')}<br>{dist_txt}"
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=popup,
            tooltip=row.get("unidade", ""),
            icon=folium.Icon(color=color),
        ).add_to(m)

    return m


def add_google_distance_time(
    df: pd.DataFrame,
    gmaps_client,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    mode: str = "driving",
    language: str = "pt-BR",
    chunk_size: int = 25,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    df = df.copy()

    # 1) filtro por raio via haversine (rápido)
    df["dist_km"] = df.apply(
        lambda r: haversine_km(center_lat, center_lon, r["latitude"], r["longitude"]),
        axis=1,
    )
    df["dentro_raio"] = df["dist_km"] <= radius_km

    # colunas google
    df["dist_google_text"] = pd.NA
    df["dist_google_m"] = pd.NA
    df["tempo_google_text"] = pd.NA
    df["tempo_google_s"] = pd.NA
    df["status_google"] = pd.NA

    idx = df.index[df["dentro_raio"] == True].tolist()
    if not idx:
        return df

    origin = (center_lat, center_lon)

    for start in range(0, len(idx), chunk_size):
        batch_idx = idx[start : start + chunk_size]
        destinations = [(float(df.at[i, "latitude"]), float(df.at[i, "longitude"])) for i in batch_idx]

        resp = gmaps_client.distance_matrix(
            origins=[origin],
            destinations=destinations,
            mode=mode,
            language=language,
            units="metric",
        )

        elements = resp["rows"][0]["elements"]

        for i, el in zip(batch_idx, elements):
            stt = el.get("status", "UNKNOWN")
            df.at[i, "status_google"] = stt

            if stt == "OK":
                dist = el["distance"]
                dur = el["duration"]
                df.at[i, "dist_google_text"] = dist.get("text")
                df.at[i, "dist_google_m"] = dist.get("value")
                df.at[i, "tempo_google_text"] = dur.get("text")
                df.at[i, "tempo_google_s"] = dur.get("value")

        time.sleep(sleep_s)

    return df


# ---------- UI ----------
st.title("📍 FATECs dentro de um raio + distância/tempo (Google Maps)")

with st.sidebar:
    st.header("Configurações")
    center_lat = st.number_input("Centro - latitude", value=-23.50787, format="%.6f")
    center_lon = st.number_input("Centro - longitude", value=-46.78395, format="%.6f")
    radius_km = st.slider("Raio (km)", min_value=5, max_value=200, value=30, step=5)

    mode = st.selectbox("Modo", ["driving", "walking", "bicycling", "transit"], index=0)
    chunk_size = st.selectbox("Chunk (destinos por request)", [10, 15, 20, 25], index=3)
    sleep_s = st.slider("Delay entre requests (s)", 0.0, 1.0, 0.1, 0.05)

st.info("Dica: o mapa e o filtro por raio funcionam sem Google. O Google entra só quando você clicar em 'Calcular rotas'.")

df = load_data("data/fatec_enderecos_geocodificados.json")

# filtro local (geodesic) — só para exibir/ordenar em linha reta
df_local = df.copy()
df_local["dist_km"] = df_local.apply(
    lambda r: geodesic((center_lat, center_lon), (r["latitude"], r["longitude"])).km,
    axis=1,
)
df_local["dentro_raio"] = df_local["dist_km"] <= radius_km

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🗺️ Mapa")
    m = build_map(df_local, center_lat, center_lon, radius_km, col_name_dist="dist_km")
    st_folium(m, height=600, width=None)

with col2:
    st.subheader("📋 Lista (ordenada pela menor distância em linha reta)")
    dentro = df_local[df_local["dentro_raio"]].sort_values("dist_km").reset_index(drop=True)
    st.write(f"Encontradas **{len(dentro)}** FATECs dentro de {radius_km} km.")
    st.dataframe(dentro[["unidade", "dist_km"]], use_container_width=True)

st.divider()

st.subheader("🚗 Distância/tempo via Google Distance Matrix (opcional)")
st.caption("Isso consome cota/billing. Use com moderação (chunk <= 25).")

if "GOOGLE_MAPS_API_KEY" not in st.secrets:
    st.warning("Defina GOOGLE_MAPS_API_KEY em st.secrets para habilitar o cálculo pelo Google.")
else:
    if st.button("Calcular rotas (Google)"):
        gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_API_KEY"])

        with st.spinner("Consultando Google Distance Matrix..."):
            df_g = add_google_distance_time(
                df=df,
                gmaps_client=gmaps,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                mode=mode,
                chunk_size=int(chunk_size),
                sleep_s=float(sleep_s),
            )

        dentro_g = (
            df_g[df_g["dentro_raio"] & (df_g["status_google"] == "OK")]
            .sort_values("dist_google_m", ascending=True)
            .reset_index(drop=True)
        )

        st.success(f"OK! {len(dentro_g)} itens com rota válida (status OK).")

        st.dataframe(
            dentro_g[["unidade", "dist_google_text", "tempo_google_text", "dist_km", "status_google"]],
            use_container_width=True,
        )

        csv = dentro_g.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar resultados (CSV)",
            data=csv,
            file_name="fatecs_raio_google.csv",
            mime="text/csv",
        )
