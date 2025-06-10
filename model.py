from tensorflow.keras import models, layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

def build_model(input_dim):
    model = models.Sequential([
        layers.InputLayer(shape=(input_dim,)),  # ← фикс тут
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    return model
