import pandas as pd
import numpy as np

def preprocess_ukr_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(columns=['title', 'addres', 'Unnamed: 0'], inplace=True, errors='ignore')

    df.rename(columns={
        'price': 'Price',
        'squer': 'Area_m2',
        'room': 'Rooms',
        'city': 'City',
        'floor': 'Floor',
        'metr': 'Metro',
        'new_rem': 'Renovated',
        'auction': 'Auction',
        'quiet_area': 'Quiet',
        'furniture': 'Furniture',
        'center': 'Center',
        'parking': 'Parking'
    }, inplace=True)


    # Числові
    for col in ['Price', 'Area_m2', 'Rooms', 'Floor',
                'Metro', 'Renovated', 'Auction', 'Quiet',
                'Furniture', 'Center', 'Parking']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # City — як string
    df['City'] = df['City'].astype(str).str.strip()


    # Удаляем пропуски
    df.dropna(inplace=True)

    return df
