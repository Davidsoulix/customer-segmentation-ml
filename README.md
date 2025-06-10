# Customer Segmentation ML Service

Microservicio de Machine Learning para segmentación de clientes usando algoritmo K-means. Analiza patrones de compra y comportamiento para clasificar automáticamente a los clientes en segmentos estratégicos.

## 🚀 Características

- **Segmentación automática** con K-means clustering
- **Análisis RFM** (Recency, Frequency, Monetary)
- **API REST** completa con FastAPI
- **Procesamiento en background** para entrenamientos largos
- **Integración con PostgreSQL** 
- **Docker** containerización completa
- **Visualización PCA** para análisis dimensional

## 📊 Segmentos Identificados

- **Champions**: Mejores clientes con alta frecuencia y valor
- **Loyal Customers**: Clientes leales con compras regulares
- **Potential Loyalists**: Clientes con potencial de fidelización
- **New Customers**: Clientes nuevos con pocas compras
- **At Risk**: Clientes en riesgo de abandono
- **Cannot Lose Them**: Clientes valiosos que requieren atención
- **Hibernating**: Clientes inactivos
- **Need Attention**: Clientes que necesitan estrategias específicas

## 🛠️ Instalación

### Opción 1: Docker (Recomendado)

```bash
# Clonar proyecto
git clone <tu-repo>
cd customer-segmentation-ml

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu configuración de base de datos

# Levantar servicios
docker-compose up -d

# Verificar que esté funcionando
curl http://localhost:8000/health
```

### Opción 2: Instalación local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export DATABASE_URL="postgresql://user:pass@localhost:5432/db"

# Ejecutar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 Uso de la API

### 1. Entrenar modelo

```bash
curl -X POST "http://localhost:8000/api/v1/segment/train" \
  -H "Content-Type: application/json" \
  -d '{"n_clusters": 5}'
```

### 2. Verificar estado del entrenamiento

```bash
curl "http://localhost:8000/api/v1/segment/status/{job_id}"
```

### 3. Obtener resultados

```bash
curl "http://localhost:8000/api/v1/segment/results/{job_id}"
```

### 4. Predecir segmento de clientes

```bash
curl -X POST "http://localhost:8000/api/v1/segment/predict" \
  -H "Content-Type: application/json" \
  -d '[1, 2, 3]'
```

### 5. Resúmenes de clusters

```bash
curl "http://localhost:8000/api/v1/segment/clusters/summary"
```

## 🧪 Pruebas

```bash
# Ejecutar script de pruebas
python test_api.py

# Pruebas unitarias
pytest tests/
```

## 📁 Estructura del Proyecto

```
customer-segmentation-ml/
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicación FastAPI principal
│   ├── config.py            # Configuración
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # Endpoints de API
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Modelos Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_service.py    # Lógica de ML
│   └── utils/
│       ├── __init__.py
│       └── database.py      # Conexión DB
├── tests/                   # Pruebas unitarias
├── data/                    # Datos de análisis
├── models/                  # Modelos entrenados
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

## 🔧 Configuración

### Variables de Entorno

```bash
# Base de datos
DATABASE_URL=postgresql://user:pass@host:port/db

# Modelo ML
MODEL_PATH=./models/
MIN_CLUSTERS=2
MAX_CLUSTERS=10
RANDOM_STATE=42

# API
API_TITLE=Customer Segmentation ML Service
API_VERSION=1.0.0
API_PREFIX=/api/v1

# CORS
ALLOWED_ORIGINS=["*"]

# Logging
LOG_LEVEL=INFO
```

## 📊 Esquema de Base de Datos

El servicio utiliza las siguientes tablas principales:

- `usuario`: Datos de clientes
- `venta`: Transacciones de venta
- `detalle_venta`: Detalles de productos vendidos
- `producto`: Catálogo de productos
- `tipo`: Categorías de productos

## 🔍 Métricas del Modelo

- **Silhouette Score**: Calidad de la segmentación
- **Inertia**: Cohesión interna de clusters
- **Características RFM**: 
  - Recency (Días desde última compra)
  - Frequency (Número de compras)
  - Monetary (Valor total gastado)

## 🚀 Deployment

### Producción con Docker

```bash
# Build imagen
docker build -t customer-segmentation-ml .

# Run container
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  --name customer-ml \
  customer-segmentation-ml
```

### Con Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: customer-seg