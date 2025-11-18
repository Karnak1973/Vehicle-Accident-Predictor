import pandas as pd
import numpy as np
import os
import glob
import sys
import json

GRID_FILE = 'data/AP7_Grid_Temporal_Y.csv'
ACCIDENTS_FILE = 'data/Accidents_AP7.csv'
METEO_FOLDER = 'data/meteo_history'
OUTPUT_FILE = 'data/AP7_Final_Training_Set.csv'
MAPPING_FILE = 'data/category_mappings.json' # Guardaremos qué significa 0, 1, 2 aquí

# Station Mapping (Highway Segment PK -> Station Code)
# Defines which weather station covers which segment of the AP-7
STATION_MAPPING = {
    0: 'X2',   # La Jonquera - Figueres
    40: 'X8',  # Girona
    100: 'WU', # Vallès / Barcelona (Castellbisbal is the representative one)
    180: 'V1', # Tarragona
    280: 'XG', # Ebro Delta
    999: 'XG'  # South limit
}

# Weather Variables
VAR_NAMES = {
    'var_32': 'temperature',
    'var_33': 'humidity',
    'var_35': 'wind_speed',
    'var_4':  'precipitation'
}

# Static Highway Features (from Accidents CSV) to propagate to the Grid
STATIC_FEATURES = [
    'C_VELOCITAT_VIA',      # Numeric (Speed Limit)
    'D_TRACAT_ALTIMETRIC',  # Categorical (Flat, Slope)
    'D_TIPUS_VIA',          # Categorical (Highway type)
    'D_SENTITS_VIA'         # Categorical (Traffic direction)
]


def load_and_prep_accidents(filepath):
    """
    Loads and cleans the original accidents CSV.
    Parses dates, hours, and PKs to match the Grid format.
    """
    print("Loading police accident data...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Date parsing
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    
    # Clean 'hora' column (handle strings like '13.2')
    if pd.api.types.is_object_dtype(df['hora']):
        df['hora'] = df['hora'].astype(str).str.replace(',', '.', regex=False)
    df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
    
    # Create timestamp
    df['timestamp_hora'] = df['data'] + pd.to_timedelta(df['hora'].round().astype(int), unit='h')
    
    # Clean 'pk' and create 'segmento_pk'
    if pd.api.types.is_object_dtype(df['pk']):
        df['pk'] = df['pk'].astype(str).str.replace(',', '.', regex=False)
    df['pk'] = pd.to_numeric(df['pk'], errors='coerce')
    
    # Segmentation logic (every 10km, same as Grid)
    df['segmento_pk'] = (df['pk'] / 10).apply(np.floor) * 10
    
    # Drop rows with missing key data
    df = df.dropna(subset=['timestamp_hora', 'segmento_pk'])
    df['segmento_pk'] = df['segmento_pk'].astype(int)
    
    return df

def generate_static_features(accidents_df):
    """
    Creates a lookup dictionary for static highway features per segment.
    Example: Segment 140 is always 'Flat' and has a limit of '120'.
    """
    print("Static Features: Learning highway characteristics per segment...")
    
    # Group by segment and take the mode (most frequent value)
    # If a segment has accidents at 80km/h and 120km/h, take the most common one.
    static_lookup = accidents_df.groupby('segmento_pk')[STATIC_FEATURES].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index()
    
    # Cleanup: Force C_VELOCITAT_VIA to be numeric, fill missing with 120.0 default
    static_lookup['C_VELOCITAT_VIA'] = pd.to_numeric(static_lookup['C_VELOCITAT_VIA'], errors='coerce').fillna(120.0)
    
    print(f" Highway features mapped for {len(static_lookup)} segments.")
    return static_lookup

def add_temporal_features(df):
    """
    Adds cyclical temporal features (Sine/Cosine transformations).
    Crucial for LSTMs to understand time continuity (23h is close to 00h).
    """
    print(" Generating cyclical temporal features...")
    
    df['hour'] = df['timestamp_hora'].dt.hour
    df['month'] = df['timestamp_hora'].dt.month
    df['dayofweek'] = df['timestamp_hora'].dt.dayofweek
    
    # Sin/Cos transformation
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Drop auxiliary columns (optional, kept for debug if needed)
    df = df.drop(columns=['hour', 'month', 'dayofweek'])
    return df

def integrate_police_overrides(final_df, accidents_df):
    """
    Adds dynamic police data (Light, Fog) ONLY where Y=1.
    Overrides weather station data if police report contradicts it.
    """
    print(" Integrating dynamic police data (Light, Fog, Real Weather Overrides)...")
    
    # Dynamic columns of interest from the accident report
    cols_dynamic = ['timestamp_hora', 'segmento_pk', 'D_CLIMATOLOGIA', 'D_BOIRA', 'D_LLUMINOSITAT', 'D_SUPERFICIE']
    
    # Merge with the main grid
    # This adds D_... columns only where an accident occurred (rest will be NaN)
    acc_slim = accidents_df[cols_dynamic].drop_duplicates(subset=['timestamp_hora', 'segmento_pk'])
    final_df = final_df.merge(acc_slim, on=['timestamp_hora', 'segmento_pk'], how='left')
    
    # If police say "Rain", force precipitation = 1.0 (if it was 0 from station)
    rain_mask = final_df['D_CLIMATOLOGIA'].str.contains('Pluja|tempesta', case=False, na=False)
    final_df.loc[rain_mask, 'precipitation'] = final_df.loc[rain_mask, 'precipitation'].apply(
        lambda x: max(x if pd.notnull(x) else 0, 1.0)
    )
    

    # Create binary column 'is_foggy'.
    # Default 0. If police say "Boira" -> 1.
    final_df['is_foggy'] = 0
    fog_mask = final_df['D_BOIRA'].str.contains('Boira', case=False, na=False)
    final_df.loc[fog_mask, 'is_foggy'] = 1
    
    # Create 'is_daylight'.
    # If police data exists, use it. Else, estimate by hour (approx 7h-20h).
    
    # Calculate by hour (base imputation)
    hour = final_df['timestamp_hora'].dt.hour
    final_df['is_daylight'] = ((hour >= 7) & (hour <= 20)).astype(int)
    
    # Correct with police data if available
    night_mask = final_df['D_LLUMINOSITAT'].str.contains('nit|fosc', case=False, na=False)
    final_df.loc[night_mask, 'is_daylight'] = 0
    
    day_mask = final_df['D_LLUMINOSITAT'].str.contains('dia|clar', case=False, na=False)
    final_df.loc[day_mask, 'is_daylight'] = 1

    # Remove original text columns to clean up (info extracted to numeric/binary)
    final_df = final_df.drop(columns=['D_CLIMATOLOGIA', 'D_BOIRA', 'D_LLUMINOSITAT'])
    
    return final_df

def load_and_unify_meteo(folder):
    """
    Loads all station CSVs, normalizes columns, and combines them.
    """
    print("Loading and unifying weather data...")
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    combined_meteo = []
    
    for filename in all_files:
        station_code = os.path.basename(filename).split('_')[1].replace('.csv', '')
        df = pd.read_csv(filename)
        df['station_id'] = station_code
        
        # Ensure all expected columns exist
        for var_col in VAR_NAMES.keys():
            if var_col not in df.columns: df[var_col] = np.nan
            
        df = df.rename(columns=VAR_NAMES)
        combined_meteo.append(df[['timestamp_hora', 'station_id'] + list(VAR_NAMES.values())])
        
    full_meteo = pd.concat(combined_meteo, ignore_index=True)
    full_meteo['timestamp_hora'] = pd.to_datetime(full_meteo['timestamp_hora'])
    return full_meteo

def assign_station_to_grid(grid_df):
    """Assigns the closest weather station to each grid segment."""
    def get_station(pk):
        sorted_limits = sorted(STATION_MAPPING.keys())
        assigned = 'XG'
        for limit in sorted_limits:
            if pk >= limit: assigned = STATION_MAPPING[limit]
            else: break
        return assigned
        
    unique_segments = pd.DataFrame({'segmento_pk': grid_df['segmento_pk'].unique()})
    unique_segments['station_id'] = unique_segments['segmento_pk'].apply(get_station)
    return grid_df.merge(unique_segments, on='segmento_pk', how='left')

def fill_missing_meteo(df):
    """
    Fills missing weather data using Forward Fill (last 24h) and Defaults.
    """
    cols_meteo = list(VAR_NAMES.values())
    df = df.sort_values(['station_id', 'timestamp_hora'])
    
    # Forward fill (limit 24h)
    df[cols_meteo] = df.groupby('station_id')[cols_meteo].ffill(limit=24)
    
    # Defaults for large gaps (e.g., missing years)
    defaults = {'temperature': 15.0, 'humidity': 60.0, 'wind_speed': 0.0, 'precipitation': 0.0}
    return df.fillna(defaults)

def apply_label_encoding(df, columns):
    """Converts categorical columns to numeric (0,1,2...) using Label Encoding.
    Nans are converted to -1"""
    print(" Applying Label Encoding...")
    mappings = {}
    
    for col in columns:
        if col in df.columns:
            # Convert to category type
            df[col] = df[col].astype('category')
            
            # Save the mapping (Index -> Category Name)
            # We map codes to strings so we can save to JSON
            mapping_dict = dict(enumerate(df[col].cat.categories))
            mappings[col] = mapping_dict
            
            # Transform column to codes (NaN becomes -1)
            df[col] = df[col].cat.codes
            
            print(f"   -> Encoded {col}: {len(mapping_dict)} categories.")
            
    return df, mappings


if __name__ == "__main__":
    print("Starting MASTER DATASET construction...")
    
    # Load Base Grid
    print(f"Reading Grid: {GRID_FILE}")
    grid_df = pd.read_csv(GRID_FILE)
    grid_df['timestamp_hora'] = pd.to_datetime(grid_df['timestamp_hora'])
    print(f"Grid loaded: {len(grid_df)} rows.")
    
    # Load and Prep Accidents (Source of static features & overrides)
    accidents_df = load_and_prep_accidents(ACCIDENTS_FILE)
    
    # STATIC FEATURE ENGINEERING (Highway)
    # Create road map and merge to grid
    static_features_df = generate_static_features(accidents_df)
    grid_df = grid_df.merge(static_features_df, on='segmento_pk', how='left')
    
    # Fill missing static features with mode (for segments with no accidents history)
    for col in STATIC_FEATURES:
        if col in grid_df.columns:
            mode_val = grid_df[col].mode()[0]
            grid_df[col] = grid_df[col].fillna(mode_val)

    # TEMPORAL FEATURE ENGINEERING (Cyclical)
    grid_df = add_temporal_features(grid_df)
    
    # METEOROLOGY (Merge + Imputation)
    meteo_df = load_and_unify_meteo(METEO_FOLDER)
    meteo_df = fill_missing_meteo(meteo_df)
    grid_df = assign_station_to_grid(grid_df)
    
    print("Merging Grid with Meteorology...")
    train_df = grid_df.merge(meteo_df, on=['timestamp_hora', 'station_id'], how='left')
    train_df = fill_missing_meteo(train_df) # Second pass for missing stations
    
    # 6. POLICE OVERRIDES (Light, Fog, Weather Correction)
    train_df = integrate_police_overrides(train_df, accidents_df)
    
    # 7. LABEL ENCODING
    
    print(" Applying One-Hot Encoding...")
    categorical_cols = ['D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA', 'D_SUPERFICIE']
    train_df, category_mappings = apply_label_encoding(train_df, categorical_cols)

    # Save mappings to JSON (Important to know what '1' means later!)
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(category_mappings, f, indent=4)
    print(f"Category mappings saved to: {MAPPING_FILE}")

    # 8. Save
    print(f" Saving MASTER DATASET to: {OUTPUT_FILE}")
    train_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n✅ PROCESS COMPLETED!")
    print(f"Dataset ready with {len(train_df)} rows and {len(train_df.columns)} variables.")
    print("Contains: Grid + Meteo + Highway Features + Light + Cyclical Time.")