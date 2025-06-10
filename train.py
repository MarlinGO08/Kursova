import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import build_model
from preprocessing import preprocess_ukr_dataset
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

def train_model():
    df = pd.read_csv("dataset.csv")
    df_clean = preprocess_ukr_dataset(df)

    X = df_clean.drop("Price", axis=1)
    y = df_clean["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обновлённые признаки
    numeric = ['Area_m2', 'Rooms', 'Floor']
    categorical = ['City', 'Metro', 'Renovated', 'Auction', 'Quiet', 'Furniture', 'Center', 'Parking']

    # Препроцессор
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Обучение модели
    model = build_model(X_train_prep.shape[1])
    history = model.fit(
        X_train_prep, y_train,
        validation_data=(X_test_prep, y_test),
        epochs=100, batch_size=8, verbose=0
    )

    y_pred = model.predict(X_test_prep)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    model.save("price_model.h5")
    joblib.dump(preprocessor, "price_preprocessor.joblib")

    # MAE графік
    mae_fig = go.Figure()
    mae_fig.add_trace(go.Scatter(
        y=history.history['val_mean_absolute_error'],
        mode='lines+markers',
        name='MAE на валідації',
        line=dict(color='royalblue')
    ))
    mae_fig.update_layout(
        title="📉 MAE по епохах",
        xaxis_title="Епоха",
        yaxis_title="MAE",
        template="plotly_dark"
    )

    # Scatter-фігура
    scatter_fig = px.scatter(
        x=y_test,
        y=y_pred.flatten(),
        labels={"x": "Фактична ціна", "y": "Прогнозована ціна"},
        title="📊 Прогноз vs Реальність",
        template="plotly_dark"
    )
    scatter_fig.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", dash="dash")
    )

    # Pie chart
    pie_fig = px.pie(
        names=["Навчання", "Тест"],
        values=[len(X_train), len(X_test)],
        title="📎 Співвідношення train/test",
        template="plotly_dark"
    )

    return mae, mse, mae_fig, scatter_fig, pie_fig, X_train, X_test
