'''
Script para preprocesar los datos de Amazon Fine Food Reviews.

Funcionalidades:
- Carga el dataset desde data/raw/Reviews.csv.
- Limpia el texto (minúsculas, elimina HTML, puntuación, números).
- Elimina stopwords.
- Realiza lematización.
- Crea una columna de sentimiento (positivo/negativo) a partir de la puntuación.
- Divide los datos en conjuntos de entrenamiento (80%), validación (10%) y prueba (10%).
- Guarda los conjuntos de datos resultantes en la carpeta data/processed/.
'''

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from sklearn.model_selection import train_test_split

# --- Descarga de recursos de NLTK ---
def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK si no están presentes."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Recurso NLTK '{name}' no encontrado. Descargando...")
            nltk.download(name)

# --- Inicialización de componentes de preprocesamiento ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- Funciones de preprocesamiento ---
def clean_text(text):
    """Limpia el texto: minúsculas, sin HTML, sin puntuación, sin números y sin espacios extra."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # a minúsculas
    text = re.sub(r'<.*?>', '', text)  # elimina tags HTML
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # elimina puntuación
    text = re.sub(r"\d+", "", text)  # elimina números
    text = re.sub(r"\s+", " ", text).strip()  # elimina espacios extra
    return text

def remove_stopwords(text):
    """Elimina stopwords del texto."""
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    """Realiza lematización en el texto."""
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_pipeline(text):
    """Aplica la pipeline completa de preprocesamiento a un texto."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# --- Script principal ---
if __name__ == "__main__":
    print("Iniciando script de preprocesamiento y división de datos...")

    # --- Descarga de recursos de NLTK ---
    download_nltk_resources()

    # --- Carga de datos ---
    raw_data_path = "data/raw/Reviews.csv"
    processed_data_path = "data/processed"

    if not os.path.exists(raw_data_path):
        print(f"Error: El archivo '{raw_data_path}' no se encuentra.")
        print("Por favor, ejecute primero el script 'src/download_data.py' de la Fase 0.")
        exit()

    print(f"Cargando datos desde '{raw_data_path}'...")
    df = pd.read_csv(raw_data_path)
    df.dropna(subset=['Text', 'Summary', 'Score'], inplace=True)
    df.drop_duplicates(subset=['UserId', 'Time', 'Text'], inplace=True)

    # --- Creación de la clase de sentimiento ---
    # Mapeamos Score a sentimiento: 1,2 -> negativo (0), 4,5 -> positivo (1). Ignoramos neutrales (3).
    df = df[df['Score'] != 3]
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

    # --- Aplicación del preprocesamiento ---
    # Usaremos una muestra para agilizar la ejecución en desarrollo.
    # Para procesar el dataset completo, comenta o aumenta el valor de n.
    sample_size = 50000 # Aumentado para tener suficientes datos para el split
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42)
    
    print(f"Aplicando preprocesamiento a la columna 'Text' en {len(df_sample)} muestras...")
    df_sample['processed_text'] = df_sample['Text'].apply(preprocess_pipeline)

    # --- División del dataset ---
    print("Dividiendo los datos en conjuntos de entrenamiento, validación y prueba...")
    train_df, temp_df = train_test_split(
        df_sample, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_sample['sentiment']
    )

    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, # 50% de 20% es 10% del total
        random_state=42, 
        stratify=temp_df['sentiment']
    )

    # --- Guardado de datos procesados ---
    os.makedirs(processed_data_path, exist_ok=True)
    
    train_path = os.path.join(processed_data_path, "train.csv")
    val_path = os.path.join(processed_data_path, "validation.csv")
    test_path = os.path.join(processed_data_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nPreprocesamiento y división de datos completados exitosamente.")
    print(f"Datos de entrenamiento guardados en: {train_path} ({len(train_df)} filas)")
    print(f"Datos de validación guardados en: {val_path} ({len(val_df)} filas)")
    print(f"Datos de prueba guardados en: {test_path} ({len(test_df)} filas)")

    # --- Ejemplo de salida ---
    print(
"--- Ejemplo de Salida (Train Data) ---\")
    print(train_df[['Text', 'processed_text', 'sentiment']].head())
    print(\"------------------------\")

