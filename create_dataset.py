import pandas as pd
import numpy as np
import os
import glob
import json
import sys

#! --- CONSTANTES ---
ACCIDENTS_FILE = 'data/Accidents_Route.csv' # Changed from Accidents_AP7.csv
METEO_FOLDER = 'data/meteo_history'
OUTPUT_FILE = 'data/AP7_Final_Training_Set.csv' # Keeping original name to avoid breaking downstream scripts
MAPPING_FILE = 'data/category_mappings.json'
SEGMENT_SIZE_KM = 10

# Station Mapping
# Route: Valencia (0) -> Vera (470)
STATION_MAPPING = {
    40: 'VALENCIA',    # 0-40 -> Valencia
    110: 'REQUENA',    # 40-110 -> Requena
    170: 'HONRUBIA',   # 110-170 -> Honrubia
    210: 'LARODA',     # 170-210 -> La Roda
    255: 'ALBACETE',   # 210-255 -> Albacete
    325: 'HELLIN',     # 255-325 -> Hellin
    400: 'MURCIA',     # 325-400 -> Murcia
    450: 'LORCA',      # 400-450 -> Lorca
    999: 'VERA'        # 450+ -> Vera
}

VAR_NAMES = {
    'var_32': 'temperature', 'var_33': 'humidity',
    'var_35': 'wind_speed', 'var_4':  'precipitation'
}

STATIC_FEATURES = [
    'C_VELOCITAT_VIA', 'D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA'
]

# --- 1. CARGA Y LIMPIEZA ---
def load_and_prep_accidents(filepath):
    print("Loading accident data...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        sys.exit(f"Error: File not found: {filepath}")

    # Limpieza de fechas y horas
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    
    # Handle synthetic time format "H.M"
    def parse_time(h_str):
        try:
            parts = str(h_str).split('.')
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return pd.Timedelta(hours=h, minutes=m)
        except:
            return pd.Timedelta(0)

    df['hora_timedelta'] = df['hora'].apply(parse_time)
    df['timestamp_hora'] = df['data'] + df['hora_timedelta']
    
    # Round timestamp to nearest hour for merging with weather
    df['timestamp_hora'] = df['timestamp_hora'].dt.round('h')

    # Limpieza de PKs y creación de Segmentos
    if pd.api.types.is_object_dtype(df['pk']):
        df['pk'] = df['pk'].astype(str).str.replace(',', '.', regex=False)
    df['pk'] = pd.to_numeric(df['pk'], errors='coerce')
    
    # Lógica unificada de segmentos (cada 10km)
    df['segmento_pk'] = (df['pk'] / SEGMENT_SIZE_KM).apply(np.floor) * SEGMENT_SIZE_KM
    
    df = df.dropna(subset=['timestamp_hora', 'segmento_pk'])
    df['segmento_pk'] = df['segmento_pk'].astype(int)
    
    return df

# --- 2. GENERACIÓN DEL GRID ---
def create_full_grid(df_accidents):
    print("Generating Temporal-Spatial Grid...")
    min_date = df_accidents['timestamp_hora'].min().floor('D')
    max_date = df_accidents['timestamp_hora'].max().ceil('D')
    
    # Explicitly set min_pk to 0 and max_pk to cover the full route
    min_pk = 0
    max_pk = 470
    
    all_hours = pd.date_range(start=min_date, end=max_date, freq='h')
    pk_segments = np.arange(min_pk, max_pk + SEGMENT_SIZE_KM, SEGMENT_SIZE_KM).astype(int)
    
    # Producto cartesiano
    grid_index = pd.MultiIndex.from_product([all_hours, pk_segments], names=['timestamp_hora', 'segmento_pk'])
    grid_df = pd.DataFrame(index=grid_index).reset_index()
    
    # Mapear Variable Objetivo (Y)
    print("Mapping accidents (Y=1)...")
    df_accidents_slim = df_accidents[['timestamp_hora', 'segmento_pk']].drop_duplicates()
    df_accidents_slim['Y_ACCIDENT'] = 1
    
    grid_df = grid_df.merge(df_accidents_slim, on=['timestamp_hora', 'segmento_pk'], how='left')
    grid_df['Y_ACCIDENT'] = grid_df['Y_ACCIDENT'].fillna(0).astype(int)
    
    print(f"Grid created: {len(grid_df)} rows.")
    return grid_df

# --- 3. INGENIERÍA DE CARACTERÍSTICAS ---
def generate_static_features(accidents_df):
    """Extrae características fijas de la carretera (tipo de vía, velocidad)"""
    print("Mapping static highway features...")
    # Synthetic data check
    if 'C_VELOCITAT_VIA' not in accidents_df.columns:
        accidents_df['C_VELOCITAT_VIA'] = 120 # Default

    static_lookup = accidents_df.groupby('segmento_pk')[STATIC_FEATURES].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index()
    # Fill defaults
    static_lookup['C_VELOCITAT_VIA'] = pd.to_numeric(static_lookup['C_VELOCITAT_VIA'], errors='coerce').fillna(120.0)
    return static_lookup

def add_temporal_features(df):
    """Seno/Coseno para horas y días"""
    print("Adding cyclical temporal features...")
    df['hour'] = df['timestamp_hora'].dt.hour
    df['month'] = df['timestamp_hora'].dt.month
    df['dayofweek'] = df['timestamp_hora'].dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df.drop(columns=['hour', 'month', 'dayofweek'])

# --- 4. METEOROLOGÍA ---
def process_meteorology(grid_df, meteo_folder):
    print("Processing Meteorology...")
    
    # 1. Unificar archivos
    all_files = glob.glob(os.path.join(meteo_folder, "*.csv"))
    combined_meteo = []
    for filename in all_files:
        station_code = os.path.basename(filename).split('_')[1].replace('.csv', '')
        df = pd.read_csv(filename)
        df['station_id'] = station_code
        # Asegurar columnas
        for var_col in VAR_NAMES.keys():
            if var_col not in df.columns: df[var_col] = np.nan
        df = df.rename(columns=VAR_NAMES)
        combined_meteo.append(df[['timestamp_hora', 'station_id'] + list(VAR_NAMES.values())])
        
    full_meteo = pd.concat(combined_meteo, ignore_index=True)
    full_meteo['timestamp_hora'] = pd.to_datetime(full_meteo['timestamp_hora'])
    
    # 2. Asignar estación al Grid
    def get_station(pk):
        for limit in sorted(STATION_MAPPING.keys()):
            if pk <= limit:
                return STATION_MAPPING[limit]
        return STATION_MAPPING[999]
    
    unique_segments = pd.DataFrame({'segmento_pk': grid_df['segmento_pk'].unique()})
    unique_segments['station_id'] = unique_segments['segmento_pk'].apply(get_station)
    
    grid_df = grid_df.merge(unique_segments, on='segmento_pk', how='left')
    
    # 3. Merge y Relleno
    # Pre-relleno meteo (ffill) para reducir huecos antes del merge
    full_meteo = full_meteo.sort_values(['station_id', 'timestamp_hora'])
    cols_meteo = list(VAR_NAMES.values())
    full_meteo[cols_meteo] = full_meteo.groupby('station_id')[cols_meteo].ffill(limit=24)
    
    merged_df = grid_df.merge(full_meteo, on=['timestamp_hora', 'station_id'], how='left')
    
    # Relleno final (defaults)
    defaults = {'temperature': 15.0, 'humidity': 60.0, 'wind_speed': 0.0, 'precipitation': 0.0}
    return merged_df.fillna(defaults)

# --- 5. OVERRIDES Y FEATURE CROSSES ---
def integrate_police_overrides(final_df, accidents_df):
    print("Integrating Police Overrides (Fog/Light)...")
    cols_dynamic = ['timestamp_hora', 'segmento_pk', 'D_CLIMATOLOGIA', 'D_BOIRA', 'D_LLUMINOSITAT']
    # Merge solo donde hubo accidentes
    acc_slim = accidents_df[cols_dynamic].drop_duplicates(subset=['timestamp_hora', 'segmento_pk'])
    final_df = final_df.merge(acc_slim, on=['timestamp_hora', 'segmento_pk'], how='left')
    
    final_df['is_foggy'] = 0
    final_df.loc[final_df['D_BOIRA'].str.contains('Boira', case=False, na=False), 'is_foggy'] = 1
    
    # Daylight calculation
    hour = final_df['timestamp_hora'].dt.hour
    final_df['is_daylight'] = ((hour >= 7) & (hour <= 20)).astype(int)
    
    final_df.loc[final_df['D_LLUMINOSITAT'].str.contains('nit|fosc', case=False, na=False), 'is_daylight'] = 0
    final_df.loc[final_df['D_LLUMINOSITAT'].str.contains('dia|clar', case=False, na=False), 'is_daylight'] = 1
    
    return final_df.drop(columns=['D_CLIMATOLOGIA', 'D_BOIRA', 'D_LLUMINOSITAT'])

def add_feature_crosses(df):
    print("Generating Feature Crosses...")
    # Asegurar orden para el cálculo de ventanas
    df = df.sort_values(['station_id', 'timestamp_hora'])
    
    # ACUMULADO DE LLUVIA (3 HORAS)
    # Si llovió hace 2 horas, el suelo sigue mojado. Esto suaviza el 0 vs 1.
    df['precip_last_3h'] = df.groupby('station_id')['precipitation'].transform(
        lambda x: x.rolling(window=3, min_periods=1).max()
    )
    
    # Variable binaria más robusta (Suelo Mojado)
    # Asumimos suelo mojado si ha llovido > 0.1mm en las últimas 3 horas
    df['wet_road'] = (df['precip_last_3h'] > 0.1).astype(int)
    
    if 'is_daylight' in df.columns:
        df['wet_and_night'] = df['wet_road'] * (1 - df['is_daylight'])
        
    # Critical segments (Hotspots) - Simulated based on synthetic data logic
    critical_segments = [0, 10, 20, 160, 170, 180, 290, 300, 310]
    if 'wind_speed' in df.columns and 'segmento_pk' in df.columns:
        df['wind_and_critical'] = df['wind_speed'] * df['segmento_pk'].isin(critical_segments).astype(int)
    
    return df

# --- 6. ENCODING Y MAIN ---
def apply_encoding(df):
    print("Applying Label Encoding...")
    mappings = {}
    cols = ['D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            mappings[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes
    
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=4)
    return df

if __name__ == "__main__":
    print("--- STARTING DATASET CREATION ---")
    
    # 1. Cargar Accidentes (Fuente de verdad)
    accidents = load_and_prep_accidents(ACCIDENTS_FILE)
    
    # 2. Crear Grid Base con Y
    master_df = create_full_grid(accidents)
    
    # 3. Pegar características estáticas del mapa
    static_feats = generate_static_features(accidents)
    master_df = master_df.merge(static_feats, on='segmento_pk', how='left')
    
    # 4. Características Temporales
    master_df = add_temporal_features(master_df)
    
    # 5. Meteorología
    master_df = process_meteorology(master_df, METEO_FOLDER)
    
    # 6. Overrides policiales (requiere el df de accidentes original para extraer texto)
    master_df = integrate_police_overrides(master_df, accidents)
    
    # 7. Feature Crosses (Tu nueva lógica)
    master_df = add_feature_crosses(master_df)
    
    # 8. Encoding final
    master_df = apply_encoding(master_df)
    
    # 9. Guardar
    print(f"Saving FINAL DATASET to {OUTPUT_FILE}...")
    master_df.to_csv(OUTPUT_FILE, index=False)
    print("✅ DONE!")
