import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kaggle_dataset(dataset_name, path='data/raw'):
    """
    Descarga y extrae un dataset de Kaggle.
    Requiere que kaggle.json esté configurado en ~/.kaggle/.
    """
    print(f"Inicializando API de Kaggle...")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(path, exist_ok=True)
    print(f"Descargando dataset '{dataset_name}' a '{path}'...")
    api.dataset_download_files(dataset_name, path=path, unzip=True)
    print(f"Dataset '{dataset_name}' descargado y extraído exitosamente.")

if __name__ == "__main__":
    # Dataset de Amazon Fine Food Reviews
    kaggle_dataset_name = 'snap/amazon-fine-food-reviews'
    download_and_extract_kaggle_dataset(kaggle_dataset_name)

    # Opcional: Cargar y mostrar las primeras filas para verificar
    try:
        df = pd.read_csv(os.path.join('data/raw', 'Reviews.csv'))
        print("\nPrimeras 5 filas del dataset:")
        print(df.head())
        print(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas.")
    except FileNotFoundError:
        print("Error: Reviews.csv no encontrado. Asegúrate de que la descarga fue exitosa.")

