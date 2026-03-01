import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_data_generators
from src.config import MODEL_PATH

def evaluate():

    _, _, test_gen = get_data_generators()

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_gen.classes

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()