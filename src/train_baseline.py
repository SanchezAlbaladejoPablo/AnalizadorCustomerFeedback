
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --- Rutas de archivos ---
PROCESSED_DATA_PATH = "data/processed"
MODELS_PATH = "models"

TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_PATH, "validation.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test.csv")

TFIDF_MODEL_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_PATH, "tfidf_logreg.pkl")

# --- Carga de datos ---
def load_data():
    """Carga los datasets de entrenamiento, validación y prueba."""
    print("Cargando datos de entrenamiento, validación y prueba...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print("Datos cargados exitosamente.")
    return train_df, val_df, test_df

# --- Entrenamiento y evaluación del modelo baseline ---
def train_and_evaluate_baseline_model(train_df, val_df, test_df):
    """
    Entrena un modelo TF-IDF + Regresión Logística y lo evalúa.
    Guarda el vectorizador TF-IDF y el modelo de Regresión Logística.
    """
    print("Iniciando entrenamiento del modelo baseline...")

    # Preparar datos
    X_train = train_df["processed_text"]
    y_train = train_df["sentiment"]
    X_val = val_df["processed_text"]
    y_val = val_df["sentiment"]
    X_test = test_df["processed_text"]
    y_test = test_df["sentiment"]

    # 1. TF-IDF Vectorizer
    print("Entrenando TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limitar a 5000 características para empezar
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF Vectorizer entrenado.")

    # 2. Logistic Regression Model
    print("Entrenando modelo de Regresión Logística...")
    log_reg_model = LogisticRegression(solver=\'liblinear\', random_state=42, class_weight=\'balanced\') # Usar class_weight para desbalance
    log_reg_model.fit(X_train_tfidf, y_train)
    print("Modelo de Regresión Logística entrenado.")

    # --- Evaluación ---
    print("Evaluando el modelo en el conjunto de validación...")
    y_pred_val = log_reg_model.predict(X_val_tfidf)
    print(f"Accuracy (Validación): {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"F1-Score (Validación): {f1_score(y_val, y_pred_val):.4f}")
    print("\nReporte de Clasificación (Validación):\n", classification_report(y_val, y_pred_val))

    print("Evaluando el modelo en el conjunto de prueba...")
    y_pred_test = log_reg_model.predict(X_test_tfidf)
    print(f"Accuracy (Prueba): {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1-Score (Prueba): {f1_score(y_test, y_pred_test):.4f}")
    print("\nReporte de Clasificación (Prueba):\n", classification_report(y_test, y_pred_test))
    print("\nMatriz de Confusión (Prueba):\n", confusion_matrix(y_test, y_pred_test))

    # --- Guardar modelos ---
    os.makedirs(MODELS_PATH, exist_ok=True)
    print(f"Guardando TF-IDF Vectorizer en {TFIDF_MODEL_PATH}...")
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"Guardando modelo de Regresión Logística en {LOGREG_MODEL_PATH}...")
    with open(LOGREG_MODEL_PATH, "wb") as f:
        pickle.dump(log_reg_model, f)
    print("Modelos guardados exitosamente.")

if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    train_and_evaluate_baseline_model(train_df, val_df, test_df)


