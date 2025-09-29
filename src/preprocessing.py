import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# --- Descarga de recursos de NLTK ---
def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK si no están presentes."""
    resources = {"corpora/stopwords": "stopwords", "corpora/wordnet": "wordnet", "corpora/omw-1.4": "omw-1.4"}
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Recurso NLTK \'{name}\' no encontrado. Descargando...")
            nltk.download(name)

download_nltk_resources()

# --- Inicialización de componentes de preprocesamiento ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- Funciones de preprocesamiento ---
def clean_text(text):
    """Limpia el texto: minúsculas, sin HTML, sin puntuación, sin números y sin espacios extra."""
    if not isinstance(text, str):
        return ""
    text = text.lower() # a minúsculas
    text = re.sub(r\'<.*?>\', \'\', text) # elimina tags HTML
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # elimina puntuación
    text = re.sub(r"\\d+", "", text) # elimina números
    text = re.sub(r"\\s+", " ", text).strip() # elimina espacios extra
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
    print("Iniciando script de preprocesamiento...")

    # --- Carga de datos ---
    raw_data_path = "data/raw/Reviews.csv"
    processed_data_path = "data/processed"
    output_file = os.path.join(processed_data_path, "processed_reviews.csv")

    if not os.path.exists(raw_data_path):
        print(f"Error: El archivo \'{raw_data_path}\' no se encuentra.")
        print("Por favor, ejecute primero el script \'src/download_data.py\' de la Fase 0.")
        exit()

    print(f"Cargando datos desde \'{raw_data_path}\'...")
    df = pd.read_csv(raw_data_path)
    df.dropna(subset=[\'Text\', \'Summary\'], inplace=True)
    df.drop_duplicates(subset=[\'UserId\', \'Time\', \'Text\'], inplace=True)

    # --- Aplicación del preprocesamiento ---
    # Usaremos una muestra para agilizar la ejecución en desarrollo.
    # Comenta la siguiente línea para procesar el dataset completo.
    df_sample = df.sample(n=10000, random_state=42)
    
    print("Aplicando preprocesamiento a la columna \'Text\' (esto puede tardar)...")
    df_sample[\'processed_text\'] = df_sample[\'Text\'].apply(preprocess_pipeline)

    print("Aplicando preprocesamiento a la columna \'Summary\'...")
    df_sample[\'processed_summary\'] = df_sample[\'Summary\'].apply(preprocess_pipeline)

    # --- Guardado de datos procesados ---
    os.makedirs(processed_data_path, exist_ok=True)
    print(f"Guardando datos procesados en \'{output_file}\'...")
    df_sample.to_csv(output_file, index=False)

    print("\\nPreprocesamiento completado exitosamente.")
    print(f"Se han procesado y guardado {len(df_sample)} filas.")

    # --- Ejemplo de salida ---
    print("\\n--- Ejemplo de Salida ---")
    pd.set_option(\'display.max_colwidth\', 200)
    print(df_sample[[\'Text\', \'processed_text\']].head())
    print("------------------------")

