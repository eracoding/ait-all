import pandas as pd

from app.utils.mappings import owner_map, brand_map


def process_input(data):
    # Features were selected for model: max_power, engine, km_driven, mileage, and year

    # Converting to the required format
    # Droping other features since our model has only 5 features
    processed_data = {
        'year': int(data.get('year', 2015)),
        'km_driven': float(data.get('km_driven', 60000)),
        'mileage': float(data.get('mileage', 19.3)),
        'engine': float(data.get('engine', 1248.0)),
        'max_power': float(data.get('max_power', 82.0))
    }

    df = pd.DataFrame([processed_data])

    return df
