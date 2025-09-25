# 📂 Customer Feedback Analyzer  

Un sistema completo de **análisis de opiniones de clientes con NLP**, que combina modelos clásicos y modernos para:  
- Clasificar reseñas en **positivas / negativas / neutras**.  
- Detectar **temas recurrentes** en grandes volúmenes de feedback.  
- Generar **resúmenes ejecutivos automáticos** para managers.  
- Exponer resultados vía **API REST (FastAPI)** y **Dashboard interactivo (Streamlit)**.  

---

## 🎯 Descripción del Proyecto  

Este proyecto implementa un pipeline profesional de **Procesamiento de Lenguaje Natural (NLP)** aplicable en contextos empresariales de e-commerce, SaaS o restauración.  

Incluye tres componentes principales:  

1. **Análisis de Sentimientos**: Comparativa entre un modelo clásico (TF-IDF + Logistic Regression) y un modelo moderno (DistilBERT fine-tuneado).  
2. **Detección de Temas**: Topic modeling con BERTopic sobre embeddings de `all-MiniLM-L6-v2`.  
3. **Resúmenes Automáticos**: Uso de `facebook/bart-large-cnn` para generar informes ejecutivos.  

---

## ✨ Características  

- 🔍 Clasificación de sentimientos (positivo, negativo, neutro)  
- 🧩 Detección automática de temas  
- 📑 Resúmenes ejecutivos  
- ⚙️ API REST con FastAPI  
- 📊 Dashboard interactivo con Streamlit  
- 🐳 Despliegue con Docker  
- 🔄 MLOps con DVC  

---

## 🔧 Tecnologías Utilizadas  

| Categoría       | Tecnologías |
|-----------------|-------------|
| Lenguaje        | Python 3.11 |
| ML clásico      | scikit-learn |
| NLP moderno     | Hugging Face (DistilBERT, BART) |
| Topic Modeling  | BERTopic |
| Backend         | FastAPI |
| Frontend        | Streamlit |
| Visualización   | matplotlib, seaborn, plotly |
| MLOps           | DVC |
| Contenedores    | Docker, docker-compose |
| CI/CD           | GitHub Actions |

---

## 📊 Dataset  

- **Fuente**: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
- **Tamaño**: 500,000 reseñas (MVP usa 100,000)  
- **Características**: texto de la reseña, rating (1–5 estrellas), ID de usuario y producto  

---

## 🚀 Instalación  

### 🔹 Prerrequisitos  
- Python 3.11  
- Git  
- Docker (opcional, recomendado)  

### 🔹 Instalación Manual  

```bash
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
🔹 Instalación con Docker
bash
Copiar código
docker-compose up --build
Esto levanta:

API en http://localhost:8000

Dashboard en http://localhost:8501

🎮 Uso
1️⃣ API REST
bash
Copiar código
cd app/api
uvicorn main:app --reload
Endpoints disponibles:

POST /predict → análisis de sentimiento

POST /topics → detección de temas

POST /summarize → resumen automático

GET /health → estado de la API

Ejemplo de request:

bash
Copiar código
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "El envío fue rápido y el producto excelente"}'
Ejemplo de respuesta:

json
Copiar código
{
  "sentiment": "positivo",
  "confidence": 0.92,
  "probabilities": {
    "negativo": 0.04,
    "neutral": 0.04,
    "positivo": 0.92
  }
}
2️⃣ Dashboard
bash
Copiar código
cd app/dashboard
streamlit run app.py
Disponible en http://localhost:8501.

Páginas incluidas:

Sentiment Analysis

Topics

Summaries

Insights

📈 Métricas
Modelo	Accuracy	F1-score
TF-IDF + Logistic Regression	~79%	~0.78
DistilBERT Fine-tuneado	~87%	~0.86

BERTopic: ~12 temas coherentes detectados (ej. “envío”, “precio”, “calidad”).

BART Summarizer: genera resúmenes ejecutivos de ~200 palabras.

🏗️ Estructura del Proyecto
css
Copiar código
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
E-commerce: detectar quejas sobre envíos o calidad

Restauración: identificar temas como “tiempo de espera”, “sabor”

SaaS: analizar tickets de soporte y problemas frecuentes

🔄 MLOps
DVC: versionado de datasets y modelos

Docker: despliegue reproducible

CI/CD: GitHub Actions para testeo y construcción automática

🧪 Testing
Ejecutar pruebas unitarias:

bash
Copiar código
pytest
📄 Licencia
Este proyecto está bajo la Licencia MIT.