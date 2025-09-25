📂 Customer Feedback Analyzer

Un sistema completo de análisis de opiniones de clientes con NLP, que combina modelos clásicos y modernos para:

Clasificar reseñas en positivas / negativas / neutras.

Detectar temas recurrentes en grandes volúmenes de feedback.

Generar resúmenes ejecutivos automáticos para managers.

Exponer resultados vía API REST (FastAPI) y Dashboard interactivo (Streamlit).

🎯 Descripción del Proyecto

Este proyecto implementa un pipeline profesional de Procesamiento de Lenguaje Natural (NLP) aplicable en contextos empresariales de e-commerce, SaaS o restauración.

Incluye tres componentes principales:

Análisis de Sentimientos: Comparativa entre un modelo clásico (TF-IDF + Logistic Regression) y un modelo moderno (DistilBERT fine-tuneado).

Detección de Temas: Topic modeling con BERTopic sobre embeddings de all-MiniLM-L6-v2.

Resúmenes Automáticos: Uso de facebook/bart-large-cnn para generar informes ejecutivos.

✨ Características Principales

🔍 Clasificación de Sentimientos: positivo, negativo, neutro.

🧩 Topic Modeling: descubre automáticamente temas frecuentes en opiniones.

📑 Resúmenes Ejecutivos: extrae los puntos clave de miles de reseñas.

⚙️ Backend API REST: endpoints listos para integrar en otros sistemas.

📊 Dashboard Interactivo: visualizaciones intuitivas para managers.

🐳 Despliegue con Docker: API y dashboard en contenedores reproducibles.

🔄 MLOps con DVC: versionado de datasets y modelos.

🔧 Tecnologías Utilizadas

Lenguaje: Python 3.11

Machine Learning Clásico: scikit-learn

NLP Moderno: Hugging Face Transformers (DistilBERT, BART)

Topic Modeling: BERTopic

Backend: FastAPI

Frontend: Streamlit

Visualización: matplotlib, seaborn, plotly

MLOps: DVC

Contenedores: Docker, docker-compose

CI/CD: GitHub Actions

📊 Dataset

Fuente: Amazon Fine Food Reviews (Kaggle)

Tamaño: 500,000 reseñas de productos → MVP usa 100,000 reseñas muestreadas.

Características: texto de la reseña, rating (1–5 estrellas), ID de usuario y producto.

🚀 Instalación y Configuración
Prerrequisitos

Python 3.11

Git

Docker (opcional, recomendado para despliegue rápido)

Instalación Manual
# Clonar el repositorio
git clone <repository-url>
cd customer_feedback_analyzer

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar recursos de NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

Instalación con Docker
docker-compose up --build


Esto levanta:

API en http://localhost:8000

Dashboard en http://localhost:8501

🎮 Uso
1. Ejecutar API REST
cd app/api
uvicorn main:app --reload


Endpoints disponibles:

POST /predict → análisis de sentimiento.

POST /topics → detección de temas.

POST /summarize → resumen automático.

GET /health → estado de la API.

Ejemplo:

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "El envío fue rápido y el producto excelente"}'


Respuesta:

{
  "sentiment": "positivo",
  "confidence": 0.92,
  "probabilities": {
    "negativo": 0.04,
    "neutral": 0.04,
    "positivo": 0.92
  }
}

2. Ejecutar Dashboard
cd app/dashboard
streamlit run app.py


Disponible en http://localhost:8501.

Páginas incluidas:

Sentiment Analysis → análisis individual de reseñas.

Topics → clustering de opiniones en temas.

Summaries → resumen automático de feedback.

Insights → gráficos empresariales.

📈 Métricas y Evaluación

Baseline LogReg (TF-IDF)

Accuracy: ~79%

F1-score: ~0.78

DistilBERT Fine-tuneado

Accuracy: ~87%

F1-score: ~0.86

BERTopic

~12 temas coherentes detectados (ej: “envío”, “precio”, “calidad”).

BART Summarizer

Genera resúmenes ejecutivos comprensibles con ~200 palabras.

🏗️ Estructura del Proyecto
customer_feedback_analyzer/
├── data/                  
│   ├── raw/              
│   └── processed/        
├── models/                
│   ├── tfidf_logreg.pkl
│   ├── distilbert_sentiment/
│   ├── bertopic_model/
│   └── bart_summarizer/
├── notebooks/             
│   ├── eda.ipynb
│   ├── baseline.ipynb
│   └── transformers.ipynb
├── src/                   
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── train_topics.py
│   ├── predict.py
│   └── utils.py
├── app/                   
│   ├── api/
│   │   └── main.py        
│   └── dashboard/
│       └── app.py         
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── dvc.yaml
├── requirements.txt
└── README.md

🌐 Casos de Uso

E-commerce

Detectar quejas sobre envíos o calidad de productos.

Resumir insights clave de miles de reseñas al mes.

Restauración

Identificar temas recurrentes en opiniones (ej. “tiempo de espera”, “sabor”).

SaaS / Software

Analizar tickets de soporte.

Detectar problemas frecuentes en el producto.

🔄 MLOps

DVC: versionado de datasets y modelos.

Docker: despliegue reproducible de API + dashboard.

CI/CD: GitHub Actions para testeo y construcción automática.

🧪 Testing

Pruebas unitarias disponibles en tests/.

Ejecutar tests:

pytest

📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE.
