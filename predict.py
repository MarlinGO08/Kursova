import pandas as pd
import tensorflow as tf
import joblib

def predict_price(area_m2: float, rooms: int, floor: int,
                  city: str, metro: str = "Ні",
                  renovated: str = "Ні", auction: str = "Ні",
                  quiet: str = "Ні", furniture: str = "Ні",
                  center: str = "Ні", parking: str = "Ні") -> int:

    def yesno(val: str) -> int:
        return 1 if val.strip().lower() == 'так' else 0

    df = pd.DataFrame([{
        'Area_m2': area_m2,
        'Rooms': rooms,
        'Floor': floor,
        'City': city.strip(),
        'Metro': yesno(metro),
        'Renovated': yesno(renovated),
        'Auction': yesno(auction),
        'Quiet': yesno(quiet),
        'Furniture': yesno(furniture),
        'Center': yesno(center),
        'Parking': yesno(parking),
        'Price': None
    }])

    model = tf.keras.models.load_model("price_model.h5")
    preprocessor = joblib.load("price_preprocessor.joblib")
    X_input = preprocessor.transform(df.drop(columns=["Price"]))
    prediction = model.predict(X_input)

    return int(prediction[0][0])
