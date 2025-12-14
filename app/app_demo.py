import streamlit as st #! pip install streamlit streamlit-folium folium joblib pandas numpy
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import joblib
import json
import datetime
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#! EJECUTAR CON ESTE COMANDO: streamlit run app/app_demo.py (NO COMO PYTHON NORMAL)

# --- CONFIGURACIÓN DE RUTAS ---
MODEL_PATH = 'models/accident_xgboost.pkl'
MAPPINGS_PATH = 'data/category_mappings.json'
GEOMETRY_PATH = 'data/route_geometry.geojson' # Actualizado al nuevo archivo

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Sistema Predicción Accidentes Ruta A3-A31-A30-A7",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric label { font-size: 1.1rem !important; }
    .stMetric .css-1wivap2 { font-size: 2rem !important; font-weight: bold; }
    h1, h2, h3 { color: #0e1117; }
    .risk-high { color: #ff2b2b; font-weight: bold; }
    .risk-med { color: #ffa500; font-weight: bold; }
    .risk-low { color: #008000; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- CARGA DE RECURSOS ---
@st.cache_resource
def load_resources():
    model = None
    mappings = None
    geometry_points = None
    geojson_layer = None

    try:
        # Cargar Modelo
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        
        # Cargar Mappings
        if os.path.exists(MAPPINGS_PATH):
            with open(MAPPINGS_PATH, 'r') as f:
                mappings = json.load(f)

        # Cargar Geometría
        if os.path.exists(GEOMETRY_PATH):
            with open(GEOMETRY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                geojson_layer = data
                
                # Extraemos y aplanamos todas las coordenadas para interpolación
                all_coords = []
                for feature in data['features']:
                    geom = feature.get('geometry', {})
                    if geom.get('type') == 'LineString':
                        coords = geom.get('coordinates', [])
                        all_coords.extend(coords)
                
                if all_coords:
                    valencia = np.array([-0.3763, 39.4699])
                    pass

    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
    
    return model, mappings, None, geojson_layer

# --- DATOS ESTÁTICOS DE TRAMOS ---
def get_static_segment_data(geometry_points=None):
    segments = []
    
    # Ruta: Valencia (0) -> Honrubia (150) -> Albacete (230) -> Murcia (370) -> Vera (470)
    # Total ~470 km
    step_km = 10
    max_pk = 470
    num_segments = int(max_pk / step_km) 
    
    # Puntos Clave (Lat, Lon)
    # Valencia: 39.46, -0.37 (PK 0)
    # Honrubia: 39.60, -2.28 (PK 150)
    # Albacete: 38.99, -1.85 (PK 230)
    # Murcia: 37.99, -1.13 (PK 370)
    # Vera: 37.24, -1.87 (PK 470)

    waypoints = [
        (0, 39.4699, -0.3763),
        (150, 39.6050, -2.2850),
        (230, 38.9944, -1.8584),
        (370, 37.9922, -1.1307),
        (470, 37.2472, -1.8710)
    ]

    def interpolate_coords(pk):
        # Encontrar a qué segmento pertenece este PK
        for i in range(len(waypoints) - 1):
            pk1, lat1, lon1 = waypoints[i]
            pk2, lat2, lon2 = waypoints[i+1]

            if pk1 <= pk <= pk2:
                ratio = (pk - pk1) / (pk2 - pk1)
                lat = lat1 + (lat2 - lat1) * ratio
                lon = lon1 + (lon2 - lon1) * ratio
                return lat, lon
        return waypoints[-1][1], waypoints[-1][2]

    for i in range(num_segments):
        pk = i * step_km
        lat, lon = interpolate_coords(pk + step_km/2) # Centro del segmento
        
        # Asignación de atributos
        velocidad = 120.0
        tipo_via = 1 
        trazado = 0  
        sentido = 1  
        
        segments.append({
            'segmento_pk': pk,
            'lat': lat,
            'lon': lon,
            'nombre_tramo': f"PK {pk}-{pk + step_km}",
            'C_VELOCITAT_VIA': velocidad,
            'D_TRACAT_ALTIMETRIC': trazado,
            'D_TIPUS_VIA': tipo_via,
            'D_SENTITS_VIA': sentido,
        })
    return pd.DataFrame(segments)

# --- FUNCIÓN DE PREDICCIÓN REAL ---
def predict_risk_real(model, df_segments, clima, hora, fecha):
    try:
        # VARIABLES TEMPORALES
        hour_sin = np.sin(2 * np.pi * hora / 24)
        hour_cos = np.cos(2 * np.pi * hora / 24)

        month = fecha.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        dayofweek = fecha.weekday()
        dow_sin = np.sin(2 * np.pi * dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * dayofweek / 7)

        X = df_segments.copy()

        # METEOROLOGÍA SIMULADA (Para Demo)
        temperature = 15.0
        if 11 <= month or month <= 2: temperature = 8.0
        if 6 <= month <= 8: temperature = 28.0

        humidity = 80.0 if clima['niebla'] or clima['lluvia'] else 50.0
        precipitation = 2.5 if clima['lluvia'] else 0.0
        wind_speed = 15.0 if clima['viento'] else 5.0

        is_foggy = 1 if clima['niebla'] else 0
        is_daylight = 1 if clima['luz'] else 0

        # Nuevas features
        precip_last_3h = precipitation
        wet_road = 1 if precipitation > 0 else 0
        wet_and_night = wet_road * (1 - is_daylight)

        # Simulación Zonas de Viento (ej. Alrededor de Albacete)
        wind_critical_segments = [160, 170, 180, 290, 300]
        wind_and_critical = [
            wind_speed if pk in wind_critical_segments else 0
            for pk in X['segmento_pk']
        ]

        #  AÑADIMOS TODAS LAS VARIABLES
        X['hour_sin'] = hour_sin
        X['hour_cos'] = hour_cos
        X['month_sin'] = month_sin
        X['month_cos'] = month_cos
        X['dow_sin'] = dow_sin
        X['dow_cos'] = dow_cos
        X['temperature'] = temperature
        X['humidity'] = humidity
        X['wind_speed'] = wind_speed
        X['precipitation'] = precipitation
        X['is_foggy'] = is_foggy
        X['is_daylight'] = is_daylight
        X['precip_last_3h'] = precip_last_3h
        X['wet_road'] = wet_road
        X['wet_and_night'] = wet_and_night
        X['wind_and_critical'] = wind_and_critical # Renombrado de wind_and_ebre


        expected_cols = [
            'segmento_pk',
            'C_VELOCITAT_VIA', 'D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'temperature', 'humidity', 'wind_speed', 'precipitation',
            'is_foggy', 'is_daylight', 'precip_last_3h', 'wet_road', 'wet_and_night',
            'wind_and_critical'
        ]

        # Verificar features del modelo
        model_feats = model.get_booster().feature_names

        X = X[expected_cols].astype(float)

        # PREDICCIÓN
        probs = model.predict_proba(X)[:, 1]

        # AJUSTE POR CONDICIONES ADVERSAS
        factor_correccion = 1.0
        if clima['lluvia']: factor_correccion += 0.30
        if clima['niebla']: factor_correccion += 0.25
        if clima['viento']: factor_correccion += 0.10
            
        probs = probs * factor_correccion
        probs = np.clip(probs, 0, 1.0)

        return probs

    except Exception as e:
        st.error(f"ERROR en predict_risk_real(): {e}")
        # st.write("Features del modelo:", model.get_booster().feature_names) # Debug
        st.stop()

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("Panel de Control")
    st.markdown("**Ruta: Valencia - Honrubia - Albacete - Murcia - Vera**")
    st.markdown("---")
    
    fecha = st.date_input("Fecha Predicción", datetime.date.today())
    hora = st.slider("Hora del día", 0, 23, datetime.datetime.now().hour, format="%dh")
    
    st.markdown("### 🌦️ Meteorología")
    col1, col2 = st.columns(2)
    with col1:
        lluvia = st.toggle("Lluvia", value=False)
        viento = st.toggle("Viento Fuerte", value=False)
    with col2:
        niebla = st.toggle("Niebla", value=False)
        is_day = 7 <= hora <= 20
        luz = st.toggle("Luz de día", value=is_day)
    
    st.markdown("### 🛣️ PK a visualizar")
    pk_min = 0
    pk_max = 470
    rango_pk = st.slider("Selecciona el rango de PK",
                         min_value=pk_min,
                         max_value=pk_max,
                         value=(pk_min, pk_max),
                         step=10,
                         format="PK %d")

    clima_dict = {'lluvia': lluvia, 'viento': viento, 'niebla': niebla, 'luz': luz}
    st.markdown("---")

# --- LÓGICA MAIN ---
model, mappings, _, geojson_layer = load_resources()

if model is not None:
    df_tramos = get_static_segment_data()
    
    # 1. PREDICCIÓN ACTUAL
    riesgos_actuales = predict_risk_real(model, df_tramos, clima_dict, hora, fecha)

    if len(riesgos_actuales) > 0:
        df_tramos['probabilidad'] = riesgos_actuales

        # Filtro PK
        pk_lower, pk_upper = rango_pk
        df_tramos = df_tramos[(df_tramos['segmento_pk'] >= pk_lower) & (df_tramos['segmento_pk'] <= pk_upper)]
        
        # Escala de Color
        norm = mcolors.Normalize(vmin=0.0, vmax=0.6)
        cmap = mcolors.LinearSegmentedColormap.from_list("RdYlGn_r",cm.RdYlGn_r(np.linspace(0, 1, 256)))
        
        def get_color(p):
            return mcolors.to_hex(cmap(norm(p)))

        df_tramos['color'] = df_tramos['probabilidad'].apply(get_color)

        # Dashboard Header
        st.title("🚔 Sistema de Predicción de Riesgo Vial")
        st.markdown(f"**Ruta: A-3 / A-31 / A-30 / A-7**")
        st.markdown(f"**Predicción para:** {fecha.strftime('%d/%m/%Y')} a las **{hora}:00h**")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        riesgo_medio = df_tramos['probabilidad'].mean() * 100
        alerts = len(df_tramos[df_tramos['probabilidad'] > 0.10])
        
        with col1: st.metric("Riesgo Global",f"{riesgo_medio:.1f}%",delta=("Alto" if riesgo_medio > 27 else ("Normal" if riesgo_medio > 12 else "Bajo")),delta_color="inverse" if riesgo_medio > 27 else ("normal" if riesgo_medio > 12 else "off"))
        with col2: st.metric("Alertas Activas", alerts, delta_color="inverse")
        with col3: st.metric("Meteorología", "Adversa" if (lluvia or niebla) else "Favorable")
        with col4: st.metric("Tráfico", "Hora Punta" if 7 <= hora <= 19 else "Fluido")

        # Mapa y Lista
        col_map, col_list = st.columns([2, 1])
        with col_map:
            st.subheader("🗺️ Mapa de Calor")

            # Centrar mapa en Albacete aprox
            m = folium.Map(location=[38.9, -1.85], zoom_start=7, tiles="CartoDB positron")

            # Dibujar trazado real (geojson)
            if geojson_layer:
                folium.GeoJson(
                    geojson_layer,
                    name="Trazado Ruta",
                    style_function=lambda x: {
                        'color': '#888888', 
                        'weight': 3,
                        'opacity': 0.4
                    }
                ).add_to(m)

            # Dibujar tramos con riesgo (circulos interpolados)
            for _, row in df_tramos.iterrows():
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=4000,
                    color=row['color'],
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"<b>{row['nombre_tramo']}</b><br>Riesgo: {row['probabilidad']:.2%}"
                ).add_to(m)
            
            st_folium(m, width="100%", height=500)

        with col_list:
            st.subheader("⚠️ Top Alertas")
            top = df_tramos.sort_values('probabilidad', ascending=False).head(5)
            for _, row in top.iterrows():
                prob = row['probabilidad'] * 100
                st.markdown(f"**{row['nombre_tramo']}**")
                st.progress(min(int(prob * 3), 100))
                st.caption(f"Probabilidad: {prob:.2f}%")

        # --- GRÁFICO TEMPORAL REAL (24 HORAS) ---
        st.markdown("---")
        st.subheader("Evolución del Riesgo (Próximas 24 Horas)")
        
        with st.spinner("Calculando previsión futura..."):
            future_risks = []
            future_hours = []
            
            base_datetime = datetime.datetime.combine(fecha, datetime.time(hora))
            
            for i in range(24):
                future_dt = base_datetime + datetime.timedelta(hours=i)
                f_hour = future_dt.hour
                f_date = future_dt.date()
                
                clima_futuro = clima_dict.copy()
                clima_futuro['luz'] = (7 <= f_hour <= 20)
                
                p_future = predict_risk_real(model, df_tramos, clima_futuro, f_hour, f_date)
                
                if len(p_future) > 0:
                    avg_risk = np.mean(p_future) * 100
                    future_risks.append(avg_risk)
                    future_hours.append(future_dt)
            
            if future_risks:
                chart_df = pd.DataFrame({'Hora': future_hours, 'Riesgo Medio (%)': future_risks})
                chart_df = chart_df.sort_values('Hora')
                st.line_chart(chart_df, x='Hora', y='Riesgo Medio (%)', color="#ff4b4b")
else:
    st.warning("El modelo aún no ha sido entrenado. Ejecuta `train_xgboost.py`.")
