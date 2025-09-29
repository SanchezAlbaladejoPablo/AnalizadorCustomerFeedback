
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# --- Rutas de archivos ---
PROCESSED_DATA_PATH = "data/processed"
MODELS_PATH = "models"

TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_PATH, "validation.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test.csv")

DISTILBERT_MODEL_DIR = os.path.join(MODELS_PATH, "distilbert_sentiment")

# --- Carga de datos ---
def load_data():
    """Carga los datasets de entrenamiento, validación y prueba."""
    print("Cargando datos de entrenamiento, validación y prueba para DistilBERT...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print("Datos cargados exitosamente.")
    return train_df, val_df, test_df

# --- Preparación del dataset para Transformers ---
class AmazonReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- Entrenamiento y evaluación de DistilBERT ---
def train_and_evaluate_distilbert(train_df, val_df, test_df):
    """
    Realiza fine-tuning de DistilBERT para clasificación de sentimiento y lo evalúa.
    Guarda el modelo fine-tuned.
    """
    print("Iniciando fine-tuning de DistilBERT...")

    # Cargar tokenizer y modelo pre-entrenado
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Tokenizar datos
    train_encodings = tokenizer(list(train_df["processed_text"]), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(list(val_df["processed_text"]), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_df["processed_text"]), truncation=True, padding=True, max_length=512)

    # Crear datasets de PyTorch
    train_dataset = AmazonReviewDataset(train_encodings, list(train_df["sentiment"]))
    val_dataset = AmazonReviewDataset(val_encodings, list(val_df["sentiment"]))
    test_dataset = AmazonReviewDataset(test_encodings, list(test_df["sentiment"]))

    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=".",  # Directorio de salida para checkpoints
        num_train_epochs=3,  # Número de épocas de entrenamiento
        per_device_train_batch_size=16,  # Tamaño de batch por dispositivo durante el entrenamiento
        per_device_eval_batch_size=16,   # Tamaño de batch por dispositivo durante la evaluación
        warmup_steps=500,  # Número de pasos para el calentamiento del learning rate
        weight_decay=0.01,  # Decaimiento de peso para regularización L2
        logging_dir=".",  # Directorio para logs de TensorBoard
        logging_steps=100, # Frecuencia de logging
        evaluation_strategy="epoch", # Evaluar al final de cada época
        save_strategy="epoch", # Guardar checkpoint al final de cada época
        load_best_model_at_end=True, # Cargar el mejor modelo al final del entrenamiento
        metric_for_best_model="f1", # Métrica para determinar el mejor modelo
        greater_is_better=True, # Un F1-score mayor es mejor
    )

    # Definir función de métricas
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Entrenar el modelo
    print("Entrenando el modelo DistilBERT...")
    trainer.train()
    print("Entrenamiento de DistilBERT completado.")

    # Evaluar el modelo en el conjunto de prueba
    print("Evaluando el modelo en el conjunto de prueba...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_dataset.labels

    print(f"Accuracy (Prueba): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score (Prueba): {f1_score(y_true, y_pred):.4f}")
    print("\nReporte de Clasificación (Prueba):\n", classification_report(y_true, y_pred))
    print("\nMatriz de Confusión (Prueba):\n", confusion_matrix(y_true, y_pred))

    # Guardar el modelo fine-tuned
    os.makedirs(DISTILBERT_MODEL_DIR, exist_ok=True)
    print(f"Guardando el modelo DistilBERT fine-tuned en {DISTILBERT_MODEL_DIR}...")
    trainer.save_model(DISTILBERT_MODEL_DIR)
    tokenizer.save_pretrained(DISTILBERT_MODEL_DIR)
    print("Modelo DistilBERT guardado exitosamente.")

if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    train_and_evaluate_distilbert(train_df, val_df, test_df)

