import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.data_loader import get_data_generators
from src.model import build_model
from src.config import MODEL_PATH

def train():

    train_gen, val_gen, test_gen = get_data_generators()

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True),
        ReduceLROnPlateau(patience=3)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )

    return model, history

if __name__ == "__main__":
    train()