import streamlit as st #! pip install streamlit streamlit-folium folium joblib pandas numpy
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import json
import datetime
import os

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
    try:
        model = joblib.load(MODEL_PATH)
        with open(MAPPINGS_PATH, 'r') as f:
            mappings = json.load(f)
        return model, mappings
    except FileNotFoundError as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None

# --- DATOS ESTÁTICOS DE TRAMOS ---
def get_static_segment_data():
    segments = []
    lat_start, lon_start = 42.4, 2.87 
    lat_end, lon_end = 40.5, 0.5 
    num_segments = 35 
    
    for i in range(num_segments):
        pk = i * 10
        alpha = i / num_segments
        tipo_via = 1 
        trazado = 0  
        sentido = 1  
        velocidad = 120.0
        
        if 140 <= pk <= 160: velocidad = 100.0 
        if 40 <= pk <= 60: trazado = 1 
        
        segments.append({
            'segmento_pk': pk,
            'lat': lat_start * (1 - alpha) + lat_end * alpha,
            'lon': lon_start * (1 - alpha) + lon_end * alpha,
            'nombre_tramo': f"AP-7 PK {pk}-{pk+10}",
            'C_VELOCITAT_VIA': velocidad,
            'D_TRACAT_ALTIMETRIC': trazado,
            'D_TIPUS_VIA': tipo_via,
            'D_SENTITS_VIA': sentido,
        })
    return pd.DataFrame(segments)

# --- FUNCIÓN DE PREDICCIÓN REAL ---
def predict_risk_real(model, df_segments, clima, hora, fecha):
    # 1. Variables Temporales
    hour_sin = np.sin(2 * np.pi * hora / 24)
    hour_cos = np.cos(2 * np.pi * hora / 24)
    
    month = fecha.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    dayofweek = fecha.weekday()
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    # 2. Construir DataFrame X
    X_input = df_segments.copy()
    
    X_input['hour_sin'] = hour_sin
    X_input['hour_cos'] = hour_cos
    X_input['month_sin'] = month_sin
    X_input['month_cos'] = month_cos
    X_input['dow_sin'] = dow_sin
    X_input['dow_cos'] = dow_cos
    
    # Meteo (Simulación de persistencia)
    X_input['temperature'] = 15.0 
    if 11 <= month <= 2: X_input['temperature'] = 5.0 
    if 6 <= month <= 8: X_input['temperature'] = 25.0 
    
    X_input['humidity'] = 75.0 if clima['niebla'] or clima['lluvia'] else 60.0
    X_input['precipitation'] = 2.0 if clima['lluvia'] else 0.0
    X_input['wind_speed'] = 15.0 if clima['viento'] else 2.0
    X_input['is_foggy'] = 1 if clima['niebla'] else 0
    X_input['is_daylight'] = 1 if clima['luz'] else 0
    
    # Orden Exacto
    expected_cols = [
        'segmento_pk', 
        'C_VELOCITAT_VIA', 'D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'temperature', 'humidity', 'wind_speed', 'precipitation',
        'is_foggy', 'is_daylight'
    ]
    
    try:
        X_final = X_input[expected_cols]
        probs = model.predict_proba(X_final)[:, 1]
        return probs
    except Exception as e:
        return []

# --- UI SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Logo_Mossos_d%27Esquadra.svg/1200px-Logo_Mossos_d%27Esquadra.svg.png", width=100)
    st.title("🎛️ Panell de Control")
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

    clima_dict = {'lluvia': lluvia, 'viento': viento, 'niebla': niebla, 'luz': luz}
    st.markdown("---")
    st.info("ℹ️ Modifica els paràmetres per veure com canvia el risc en temps real")

# --- LÓGICA MAIN ---
model, mappings = load_resources()

if model is not None:
    df_tramos = get_static_segment_data()
    
    # 1. PREDICCIÓN ACTUAL (NOW)
    riesgos_actuales = predict_risk_real(model, df_tramos, clima_dict, hora, fecha)
    
    if len(riesgos_actuales) > 0:
        df_tramos['probabilidad'] = riesgos_actuales
        
        def get_color(p):
            if p > 0.40: return 'red'
            if p > 0.10: return 'orange'
            if p > 0.05: return 'yellow'
            return 'green'

        df_tramos['color'] = df_tramos['probabilidad'].apply(get_color)

        # Dashboard Header
        st.title("🚔 Sistema de Predicció de Risc Viari (AP-7)")
        st.markdown(f"**Predicció per a:** {fecha.strftime('%d/%m/%Y')} a les **{hora}:00h**")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        riesgo_medio = df_tramos['probabilidad'].mean() * 100
        alerts = len(df_tramos[df_tramos['probabilidad'] > 0.10])
        
        with col1: st.metric("Risc Global", f"{riesgo_medio:.1f}%", delta=f"{'⬆️ Alt' if riesgo_medio > 10 else '⬇️ Normal'}")
        with col2: st.metric("Alertes Actives", alerts, delta_color="inverse")
        with col3: st.metric("Meteorologia", "Adversa" if (lluvia or niebla) else "Favorable")
        with col4: st.metric("Trànsit", "Hora Punta" if 7 <= hora <= 19 else "Fluid")

        # Mapa y Lista
        col_map, col_list = st.columns([2, 1])
        with col_map:
            st.subheader("🗺️ Mapa de Calor")
            m = folium.Map(location=[41.5, 1.5], zoom_start=8, tiles="CartoDB positron")
            for _, row in df_tramos.iterrows():
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=4000, color=row['color'], fill=True, fill_opacity=0.7,
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
        st.subheader("📈 Evolució del Risc (Pròximes 24 Hores)")
        
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
        st.warning("Error en la predicción. Revisa los datos de entrada.")
else:
    st.error("No se ha podido cargar el modelo.")