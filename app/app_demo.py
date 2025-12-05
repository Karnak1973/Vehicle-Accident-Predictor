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

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Sistema Predicció Accidents AP-7",
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

        # Cargar Geometría (NUEVO)
        geo_path = "data/ap7_geometry.geojson"
        if os.path.exists(geo_path):
            with open(geo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                geojson_layer = data # Guardamos para pintar la línea completa luego
                
                # Extraemos y aplanamos todas las coordenadas
                all_coords = []
                for feature in data['features']:
                    geom = feature.get('geometry', {})
                    if geom.get('type') == 'LineString':
                        all_coords.extend(geom.get('coordinates', []))
                

                # Ordenamos los puntos por Latitud descendente (Norte -> Sur) para simular el avance de los PKs
                # x[1] es latitud, x[0] es longitud
                if all_coords:
                    # Filtramos Latitud > 40.52 (Frontera Cataluña-Valencia)
                    # Esto evita que la línea baje hasta Castellón/Valencia
                    catalunya_points = [c for c in all_coords if c[1] > 40.52]
                    
                    # Ordenamos Norte -> Sur
                    geometry_points = sorted(catalunya_points, key=lambda x: x[1], reverse=True)

    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
    
    return model, mappings, geometry_points, geojson_layer

# --- DATOS ESTÁTICOS DE TRAMOS ---
def get_static_segment_data(geometry_points=None):
    segments = []
    
    # Aumentamos resolución para que no haya huecos
    # De 0 a 340 km, saltando de 5 en 5 (69 tramos aprox)
    step_km = 5  
    max_pk = 340
    num_segments = int(max_pk / step_km) 
    
    # Fallbacks (por si acaso)
    lat_start_fb, lon_start_fb = 42.4, 2.87 
    lat_end_fb, lon_end_fb = 40.5, 0.5 

    for i in range(num_segments):
        pk = i * step_km  # PKs: 0, 5, 10, 15...
        
        # LÓGICA GEOMETRÍA
        if geometry_points and len(geometry_points) > 100:
            idx = int((pk / max_pk) * (len(geometry_points) - 1))
            idx = min(idx, len(geometry_points) - 1)
            lon_real, lat_real = geometry_points[idx]
        else:
            alpha = i / num_segments
            lat_real = lat_start_fb * (1 - alpha) + lat_end_fb * alpha
            lon_real = lon_start_fb * (1 - alpha) + lon_end_fb * alpha

        # Asignación de atributos (Simplificada para el ejemplo)
        velocidad = 120.0
        tipo_via = 1 
        trazado = 0  
        sentido = 1  
        
        # Ajustamos lógica de tramos especiales a los nuevos PKs
        if 140 <= pk <= 160: velocidad = 100.0 
        if 40 <= pk <= 60: trazado = 1 
        
        segments.append({
            'segmento_pk': pk,
            'lat': lat_real,
            'lon': lon_real,
            # CAMBIO: Nombre del tramo actualizado al step de 5km
            'nombre_tramo': f"AP-7 PK {pk}-{pk + step_km}",
            'C_VELOCITAT_VIA': velocidad,
            'D_TRACAT_ALTIMETRIC': trazado,
            'D_TIPUS_VIA': tipo_via,
            'D_SENTITS_VIA': sentido,
        })
    return pd.DataFrame(segments)

# --- FUNCIÓN DE PREDICCIÓN REAL ---
def predict_risk_real(model, df_segments, clima, hora, fecha):
    """
    Construye todas las features que el modelo XGBoost requiere
    y devuelve las probabilidades de accidente para cada tramo.
    """

    try:
        # VARIABLES TEMPORALES
        hour_sin = np.sin(2 * np.pi * hora / 24)
        hour_cos = np.cos(2 * np.pi * hora / 24)

        month = fecha.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        dayofweek = fecha.weekday()  # lunes=0, domingo=6
        dow_sin = np.sin(2 * np.pi * dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * dayofweek / 7)

        X = df_segments.copy()

        # METEOROLOGÍA
        # Simulacion de la temperatura segun el mes (pq no tenemos temperatura a tiempo real de meteocat: es de pago)
        temperature = 15.0
        if 11 <= month or month <= 2:
            temperature = 7.0
        if 6 <= month <= 8:
            temperature = 27.0

        humidity = 80.0 if clima['niebla'] or clima['lluvia'] else 55.0
        precipitation = 2.5 if clima['lluvia'] else 0.0
        wind_speed = 15.0 if clima['viento'] else 3.0

        is_foggy = 1 if clima['niebla'] else 0
        is_daylight = 1 if clima['luz'] else 0

        # Nuevas features usadas por tu modelo
        precip_last_3h = precipitation   # persistencia simple
        wet_road = 1 if precipitation > 0 else 0
        wet_and_night = wet_road * (1 - is_daylight)

        # Zonas especiales del Ebre → viento
        tramos_ebre = [290, 300, 310, 320, 330]
        wind_and_ebre = [
            wind_speed if pk in tramos_ebre else 0
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
        X['wind_and_ebre'] = wind_and_ebre

        # ORDEN EXACTO DE FEATURES
        expected_cols = [
            'segmento_pk',
            'C_VELOCITAT_VIA', 'D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'temperature', 'humidity', 'wind_speed', 'precipitation',
            'is_foggy', 'is_daylight', 'precip_last_3h', 'wet_road', 'wet_and_night',
            'wind_and_ebre'
        ]

        X = X[expected_cols].astype(float)

        # PREDICCIÓN
        probs = model.predict_proba(X)[:, 1]

        # AJUSTE POR CONDICIONES ADVERSAS (LLUVIA, NIEBLA, VIENTO) - Segun parametros DGT
        factor_correccion = 1.0
    
        if clima['lluvia']:
            factor_correccion += 0.30 # +30% de riesgo base por lluvia
        if clima['niebla']:
            factor_correccion += 0.25  # +25% de riesgo base por niebla
        if clima['viento']:
            factor_correccion += 0.10
            
        # Aplicamos factor pero limitamos a 1.0
        probs = probs * factor_correccion
        probs = np.clip(probs, 0, 1.0) # Que no pase del 100%

        return probs

    except Exception as e:
        st.error(f"ERROR en predict_risk_real(): {e}")
        st.write("Columnas esperadas:", expected_cols)
        st.write("Columnas reales recibidas:", list(df_segments.columns))
        st.stop()

    except Exception as e:
        st.error(f"ERROR en predict_risk_real(): {e}")
        st.write("Columnas del modelo:", model.get_booster().feature_names)
        st.write("Columnas recibidas:", list(df_tramos.columns))
        st.stop()

# --- UI SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Logo_dels_Mossos_d%27Esquadra_sense_fons.svg", width=500)
    st.title("Panell de Control")
    st.markdown("---")
    
    fecha = st.date_input("Data Predicció", datetime.date.today())
    hora = st.slider("Hora del dia", 0, 23, datetime.datetime.now().hour, format="%dh")
    
    st.markdown("### 🌦️ Meteorologia")
    col1, col2 = st.columns(2)
    with col1:
        lluvia = st.toggle("Pluja", value=False)
        viento = st.toggle("Vent Fort", value=False)
    with col2:
        niebla = st.toggle("Boira", value=False)
        is_day = 7 <= hora <= 20
        luz = st.toggle("Llum de dia", value=is_day)
    
    st.markdown("### 🛣️ Rang AP-7 a visualitzar")
    pk_min = 0
    pk_max = 340
    rango_pk = st.slider("Selecciona el rang de PK",
                         min_value=pk_min,
                         max_value=pk_max,
                         value=(pk_min, pk_max),
                         step=10,
                         format="PK %d")

    clima_dict = {'lluvia': lluvia, 'viento': viento, 'niebla': niebla, 'luz': luz}
    st.markdown("---")
    st.info("ℹ️ Modifica els paràmetres per veure com canvia el risc en temps real")

# --- LÓGICA MAIN ---
model, mappings, geometry_points, geojson_layer = load_resources()

if model is not None:
    df_tramos = get_static_segment_data(geometry_points)
    
    # 1. PREDICCIÓN ACTUAL
    riesgos_actuales = predict_risk_real(model, df_tramos, clima_dict, hora, fecha)

    df_tramos['probabilidad'] = riesgos_actuales

    if len(riesgos_actuales) > 0:
        df_tramos['probabilidad'] = riesgos_actuales

        # Filtro PK
        pk_lower, pk_upper = rango_pk
        df_tramos = df_tramos[(df_tramos['segmento_pk'] >= pk_lower) & (df_tramos['segmento_pk'] <= pk_upper)]
        
        #* --- DEFINICIÓN DE ESCALA COLOR ---
        # vmin=0.0: Riesgo nulo (Verde absoluto)
        # vmax=0.7: Riesgo muy alto (Rojo absoluto)
        # No ponemos 1 porque es muy raro que nos de el 100% de probabilidad en la vida real
        norm = mcolors.Normalize(vmin=0.0, vmax=0.6)
        cmap = mcolors.LinearSegmentedColormap.from_list("RdYlGn_r",cm.RdYlGn_r(np.linspace(0, 1, 256))) # Colores invertidos: Verde=Seguro, Rojo=Peligro
        
        def get_color(p):
            # Convierte la probabilidad (0.0 a 1.0) en un código Hexadecimal (#RRGGBB)
            return mcolors.to_hex(cmap(norm(p)))
        # def get_color(p):
        #     if p > 0.50: return 'darkred'
        #     if p > 0.40: return 'red'
        #     if p > 0.30: return 'orange'
        #     if p > 0.20: return 'yellow'
        #     if p > 0.05: return 'yellowgreen'
        #     return 'green'

        df_tramos['color'] = df_tramos['probabilidad'].apply(get_color)

        # Dashboard Header
        st.title("🚔 Sistema de Predicció de Risc Viari (AP-7)")
        st.markdown(f"**Predicció per a:** {fecha.strftime('%d/%m/%Y')} a les **{hora}:00h**")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        riesgo_medio = df_tramos['probabilidad'].mean() * 100
        alerts = len(df_tramos[df_tramos['probabilidad'] > 0.10])
        
        with col1: st.metric("Risc Global",f"{riesgo_medio:.1f}%",delta=("Alt" if riesgo_medio > 27 else ("Normal" if riesgo_medio > 12 else "Baix")),delta_color="inverse" if riesgo_medio > 27 else ("normal" if riesgo_medio > 12 else "off"))
        with col2: st.metric("Alertes Actives", alerts, delta_color="inverse")
        with col3: st.metric("Meteorologia", "Adversa" if (lluvia or niebla) else "Favorable")
        with col4: st.metric("Trànsit", "Hora Punta" if 7 <= hora <= 19 else "Fluid")

        # Mapa y Lista
        col_map, col_list = st.columns([2, 1])
        with col_map:
            st.subheader("🗺️ Mapa de Calor")

            # --- LEYENDA CSS ---
            # Creamos un gradiente visual que coincide con tu escala 'RdYlGn_r'
            legend_html = legend_html = """
            <div style="
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                margin-bottom: 15px; 
                font-family: sans-serif; 
                font-size: 0.8rem;">
                
                <div style="
                    width: 100%; 
                    height: 15px; 
                    background: linear-gradient(to right, #228B22, #9ACD32, #FFD700, #FFA500, #FF0000, #8B0000); 
                    border-radius: 5px;
                    border: 1px solid #ddd;">
                </div>
                
                <div style="
                    display: flex; 
                    justify-content: space-between; 
                    width: 100%; 
                    margin-top: 5px; 
                    color: #555;">
                    <span>Risc Baix</span>
                    <span>Moderat</span>
                    <span>Crític</span>
                </div>
            </div>
            """
            components.html(legend_html, height=50)

            m = folium.Map(location=[41.5, 1.5], zoom_start=8, tiles="CartoDB positron")
            # Dibujar trazado real de la AP-7 (linea gris)
            if geojson_layer:
                folium.GeoJson(
                    geojson_layer,
                    name="Trazado AP-7",
                    style_function=lambda x: {
                        'color': '#888888', 
                        'weight': 4, 
                        'opacity': 0.5
                    }
                ).add_to(m)
            # Dibujar tramos con riesgo (circulos)    
            for _, row in df_tramos.iterrows():
                # Filtramos para dibujar solo lo seleccionado en el slider
                if pk_lower <= row['segmento_pk'] <= pk_upper:
                    folium.Circle(
                        location=[row['lat'], row['lon']],
                        radius=4000, 
                        color=row['color'], 
                        fill=True, 
                        fill_opacity=0.8,
                        popup=f"<b>PK {row['segmento_pk']}</b><br>Risc: {row['probabilidad']:.2%}"
                    ).add_to(m)
            
            st_folium(m, width="100%", height=500)

        with col_list:
            st.subheader("⚠️ Top Alertes")
            top = df_tramos.sort_values('probabilidad', ascending=False).head(5)
            for _, row in top.iterrows():
                prob = row['probabilidad'] * 100
                st.markdown(f"**{row['nombre_tramo']}**")
                st.progress(min(int(prob * 3), 100))
                st.caption(f"Probabilitat: {prob:.2f}%")

        # --- GRÁFICO TEMPORAL REAL (24 HORAS) ---
        st.markdown("---")
        st.subheader("Evolució del Risc (Pròximes 24 Hores)")
        
        with st.spinner("Calculant previsió futura..."):
            future_risks = []
            future_hours = []
            
            # Fecha base para el cálculo (empezando en la hora seleccionada)
            base_datetime = datetime.datetime.combine(fecha, datetime.time(hora))
            
            for i in range(24):
                # Calcular fecha/hora futura
                future_dt = base_datetime + datetime.timedelta(hours=i)
                f_hour = future_dt.hour
                f_date = future_dt.date()
                
                # Ajustar luz automáticamente para el futuro (ciclo día/noche real)
                # Mantenemos lluvia/niebla constante (persistencia) pero cambiamos la luz
                clima_futuro = clima_dict.copy()
                clima_futuro['luz'] = (7 <= f_hour <= 20)
                
                # Predecir riesgo para todos los tramos en esa hora futura
                p_future = predict_risk_real(model, df_tramos, clima_futuro, f_hour, f_date)
                
                if len(p_future) > 0:
                    avg_risk = np.mean(p_future) * 100
                    future_risks.append(avg_risk)
                    future_hours.append(f"{f_hour}:00")
            
            # Crear gráfico
            if future_risks:
                chart_df = pd.DataFrame({'Hora': future_hours, 'Risc Mig (%)': future_risks})
                st.line_chart(chart_df, x='Hora', y='Risc Mig (%)', color="#ff4b4b")
                
                # Insight automático
                max_risk_h = future_hours[np.argmax(future_risks)]
                st.info(f"💡 Atenció: El pic màxim de risc s'espera a les **{max_risk_h}**.")

    else:
        st.error("Error: la predicció no coincideix amb el nombre de trams filtrats")
        st.stop()
else:
    st.error("No se ha podido cargar el modelo.")