import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Parameters for synthetic data
OUTPUT_FILE = "data/Accidents_Route.csv"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
NUM_ACCIDENTS = 5000  # Number of synthetic accidents to generate

# Route definition: Valencia (0) -> Vera (470)
# A-3: 0 - 150
# A-31: 150 - 230
# A-30: 230 - 370
# A-7: 370 - 470
TOTAL_LENGTH_KM = 470

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_synthetic_accidents():
    print(f"Generating {NUM_ACCIDENTS} synthetic accidents...")

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    data = []

    for _ in range(NUM_ACCIDENTS):
        dt = random_date(start, end)

        # Date and Time
        date_str = dt.strftime("%d/%m/%Y")
        hour_str = f"{dt.hour}.{dt.minute}"

        # PK (Location)
        # Weighted random to simulate hotspots (e.g., around cities)
        # Cities approx PK: Valencia 0, Albacete 170, Murcia 300
        if random.random() < 0.3:
            # Hotspots around cities
            pk = random.choice([
                random.uniform(0, 20),    # Valencia exit
                random.uniform(160, 180), # Albacete
                random.uniform(290, 310)  # Murcia
            ])
        else:
            pk = random.uniform(0, TOTAL_LENGTH_KM)

        pk = round(pk, 1)

        # Random Features
        weather_conditions = ["Serè", "Pluja", "Ennudesit", "Boira"]
        weather_probs = [0.7, 0.15, 0.1, 0.05]
        weather = random.choices(weather_conditions, weights=weather_probs)[0]

        light_conditions = ["Clar", "Fosc", "Penombra"]
        # Basic logic for light
        if 7 <= dt.hour <= 20:
            light = "Clar"
        else:
            light = "Fosc"

        road_types = ["Autovia", "Autovia", "Autovia", "Carretera"]
        road_type = random.choice(road_types)

        # CSV Structure matching original Accidents_AP7.csv columns used in create_dataset.py
        # Columns used: 'data', 'hora', 'pk', 'D_TRACAT_ALTIMETRIC', 'D_TIPUS_VIA', 'D_SENTITS_VIA', 'D_CLIMATOLOGIA', 'D_BOIRA', 'D_LLUMINOSITAT'

        record = {
            "data": date_str,
            "hora": hour_str,
            "pk": pk,
            "D_TRACAT_ALTIMETRIC": random.choice(["Pla", "Rampa", "Pendent"]),
            "D_TIPUS_VIA": road_type,
            "D_SENTITS_VIA": random.choice(["Un sentit", "Doble sentit"]),
            "D_CLIMATOLOGIA": weather,
            "D_BOIRA": "Boira" if weather == "Boira" else "No es de el cas",
            "D_LLUMINOSITAT": light
        }

        data.append(record)

    df = pd.DataFrame(data)

    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved synthetic accidents to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_synthetic_accidents()
