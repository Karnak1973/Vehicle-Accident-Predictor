import pandas as pd
from sodapy import Socrata #! pip install sodapy
import os
import sys
import time

# We use the portal of Open Data Catalonia
SOCRATA_DOMAIN = "analisi.transparenciacatalunya.cat"
SOCRATA_DATASET_ID = "nzvn-apee" # ID of the Meteorological Data dataset (XEMA)

APP_TOKEN = None # Token if you have one, else None (is not necessary for small requests)

# Nearby stations along AP-7 highway
# X2: Agullana, X8: Vilobí, X4: Bcn/Raval (or WU: Castellbisbal), V1: Constantí, XG: Alcanar
STATIONS = ['X2', 'X8', 'WU', 'V1', 'XG'] 

# Variables (MeteoCat Codes): 32 (Temp), 33 (Humidity), 35 (Wind), 4 (Rain)
VARIABLES_OF_INTEREST = ['32', '33', '35', '4']

YEAR_START = 2010
YEAR_END = 2025

OUTPUT_DIR = "data/meteo_history"

def fetch_station_year_by_year(station_code):
    print(f"\n--- Starting ROBUST download for Station {station_code} ---")
    
    # Set high timeout in the constructor (key to avoid cuts)
    client = Socrata(SOCRATA_DOMAIN, APP_TOKEN, timeout=60)
    
    all_years_data = []
    
    # Format variables with single quotes for SQL (Your correction)
    vars_formatted = ",".join([f"'{x}'" for x in VARIABLES_OF_INTEREST])
    
    # Loop Year by Year (Divide and Conquer)
    for year in range(YEAR_START, YEAR_END + 1):
        print(f"  Downloading year {year}...", end="", flush=True)
        
        # Filter only by current year
        where_clause = (
            f"codi_estacio='{station_code}' "
            f"AND data_lectura >= '{year}-01-01' "
            f"AND data_lectura <= '{year}-12-31' "
            f"AND codi_variable IN ({vars_formatted})"
        )
        
        try:
            results = client.get(
                SOCRATA_DATASET_ID, 
                where=where_clause,
                limit=100000, # More than enough limit for 1 year of data (8760 hours * 4 variables = ~35k rows)
                order="data_lectura ASC"
            )
            
            if results:
                df_year = pd.DataFrame.from_records(results)
                all_years_data.append(df_year)
                print(f" ✅ OK ({len(df_year)} rows)")
            else:
                print(" ⚠️ No data this year.")
            
            # Courtesy pause to avoid overloading the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f" ❌ Error in {year}: {e}")
            # We don't stop the script, we try the next year
            continue

    #  Merge all years
    if not all_years_data:
        print(f" No data has been recovered for {station_code} in any year.")
        return None
        
    print(f" Merging {len(all_years_data)} years of data...")
    full_df = pd.concat(all_years_data, ignore_index=True)
    return full_df

def process_socrata_df(df):
    print("  Processing and pivoting table...")
    
    # Basic cleaning
    df = df.dropna(subset=['data_lectura', 'codi_variable', 'valor_lectura'])

    # Convert types (optimized)
    df['valor_lectura'] = pd.to_numeric(df['valor_lectura'], errors='coerce')
    df['data_lectura'] = pd.to_datetime(df['data_lectura'])
    
    # Pivot: From rows to columns
    df_pivot = df.pivot_table(
        index='data_lectura', 
        columns='codi_variable', 
        values='valor_lectura',
        aggfunc='mean'
    )
    
    df_pivot = df_pivot.reset_index()
    
    # Rename columns dynamically
    df_pivot.columns = [f'var_{col}' if col != 'data_lectura' else 'timestamp_hora' for col in df_pivot.columns]
    
    # Round hour and final grouping
    df_pivot['timestamp_hora'] = df_pivot['timestamp_hora'].dt.round('h')
    df_final = df_pivot.groupby('timestamp_hora').mean().reset_index()
    
    return df_final

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting download from {YEAR_START} to {YEAR_END}...")
    
    for estacio in STATIONS:
        df = fetch_station_year_by_year(estacio)
        
        if df is not None and not df.empty:
            df_clean = process_socrata_df(df)
            
            output_file = os.path.join(OUTPUT_DIR, f"meteo_{estacio}.csv")
            df_clean.to_csv(output_file, index=False)
            print(f"Saved: {output_file} ({len(df_clean)} total hours)\n")
        else:
            print(f"Skipping station {estacio} (empty).\n")

    print("--- Process completed successfully ---")


"""
CODE USING METEOCAT API --> API DOESN'T WORK BECAUSE IT GIVES 403 ERROR

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime

API_KEY = "0UKnjucrjp3ePOpc7fd0O9p3kgbfdL482wTN9ARG"  

#! XEMA stations near AP-7 (ADD AS NECESSARY)
#! e.g.: X4 (El Papiol), X8 (Vilobí d'Onyar), XD (Tarragona)
STATIONS = ['X2',  # Agullana (Near La Jonquera - PK 0-20)
    'X8',  # Vilobí d'Onyar (Girona Airport - PK 50-80)
    'X9',  # Vilanova del Vallès (Granollers Area - PK 130-140)
    'X4',  # Barcelona - El Raval / Barcelona Area (Your reference for central area)
    'WU',  # Castellbisbal (AP-7 junction with A-2 - PK 160-175) - VERY IMPORTANT
    'D5',  # Vilafranca del Penedès (Alt Penedès - PK 190-200)
    'V1',  # Constantí (Tarragona/Reus - PK 240-260)
    'UA',  # L'Aldea (Ebre Delta - PK 310-320)
    'XG'   # Alcanar (Southern limit - PK 330-345)
    ]

# Get the variable codes from the API documentation
# 32 = Temperature (ºC)
# 4  = Precipitation (mm)
# 35 = Average wind speed (m/s)
# 47 = Maximum wind gust (m/s)
# 50 = Relative humidity (%)
CODIGO_VARIABLES = [32, 4, 35, 47, 50]

DATE_START = "2010-01-01"
DATE_END = "2023-12-31"
OUTPUT_DIR = "data/meteo_history"
# ---------------------

BASE_URL = "https://api.meteo.cat/xema/v1"

def generate_date_chunks(start_date, end_date, chunk_days=30):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_chunks = []
    current_start = start
    while current_start <= end:
        current_end = current_start + pd.Timedelta(days=chunk_days)
        if current_end > end: current_end = end
        date_chunks.append((current_start.strftime('%Y-%m-%dZ'), current_end.strftime('%Y-%m-%dZ')))
        current_start = current_end + pd.Timedelta(days=1)
    return date_chunks

def fetch_data_for_station(codi_estacio):
    print(f"\n--- Starting download for Station: {codi_estacio} ---")
    
    headers = {"x-api-key": API_KEY}
    
    date_chunks = generate_date_chunks(DATE_START, DATE_END)
    all_data_df = None
    
    for codi_variable in CODIGO_VARIABLES:
        print(f"  Downloading variable: {codi_variable}...")
        variable_data = []
        
        endpoint = f"{BASE_URL}/estacions/{codi_estacio}/variables/{codi_variable}/lectures"
        
        for i, (data_ini, data_fi) in enumerate(date_chunks):
            params = {"dataIni": data_ini, "dataFi": data_fi}
            
            try:
                response = requests.get(endpoint, headers=headers, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'lectures' in data and data['lectures']:
                        variable_data.extend(data['lectures'])
                    print(f"    Chunk {i+1}/{len(date_chunks)} OK.", end='\r')
                elif response.status_code == 429:
                    print("    API limit exceeded. Waiting 5s...")
                    time.sleep(5)
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"    Failed in chunk {i+1}: {e}")

        if not variable_data:
            continue
            
        df_var = pd.DataFrame(variable_data)
        df_var = df_var.rename(columns={'valor': f'var_{codi_variable}'})
        
        col_fecha = 'data' if 'data' in df_var.columns else 'dataLectura'
        if col_fecha in df_var.columns:
             df_var = df_var[[col_fecha, f'var_{codi_variable}']]
             df_var = df_var.rename(columns={col_fecha: 'raw_timestamp'})
        
        if all_data_df is None:
            all_data_df = df_var
        else:
            all_data_df = pd.merge(all_data_df, df_var, on=['raw_timestamp'], how='outer')

    return all_data_df

def process_timestamp(df):
    if df is None or df.empty: return None
    df['timestamp_hora'] = pd.to_datetime(df['raw_timestamp']).dt.tz_localize(None)
    df['timestamp_hora'] = df['timestamp_hora'].dt.round('h')
    df = df.drop(columns=['raw_timestamp'], errors='ignore')
    df = df.groupby('timestamp_hora').mean().reset_index()
    return df

if __name__ == "__main__":
    # Security check for API key
    if "AQUI_TU" in API_KEY:
        print("ERROR: You must set your MeteoCat API key in the script before running.")
        sys.exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for estacio in STATIONS:
        df = fetch_data_for_station(estacio)
        if df is not None:
            df_clean = process_timestamp(df)
            if df_clean is not None:
                filename = f"{OUTPUT_DIR}/meteo_{estacio}.csv"
                df_clean.to_csv(filename, index=False)
                print(f"Saved: {filename}")
            else:
                print(f"Station {estacio} empty.")
        else:
            print(f"Station {estacio} failed to download data.")
"""