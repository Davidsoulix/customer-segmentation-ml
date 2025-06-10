from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from loguru import logger
import sys
from datetime import datetime

from app.config import settings
from app.api.routes import router
from app.services.ml_service import ml_service

# Configurar logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Microservicio de Machine Learning para segmentación de clientes usando K-means",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas de la API
app.include_router(router, prefix=settings.API_PREFIX, tags=["segmentation"])

@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación
    """
    logger.info("🚀 Iniciando Customer Segmentation ML Service")
    logger.info(f"📊 Configuración: Min clusters={settings.MIN_CLUSTERS}, Max clusters={settings.MAX_CLUSTERS}")
    
    # Intentar cargar modelo existente
    try:
        ml_service.load_model()
        logger.info("✅ Modelo ML cargado exitosamente")
    except Exception as e:
        logger.warning(f"⚠️  No se pudo cargar modelo existente: {e}")
        logger.info("🔧 Se entrenará un nuevo modelo en la primera solicitud")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento de cierre de la aplicación
    """
    logger.info("🛑 Cerrando Customer Segmentation ML Service")

@app.get("/")
async def root():
    """
    Endpoint raíz con información del servicio
    """
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "api_docs": "/api/v1/docs",
        "description": "Microservicio de ML para segmentación de clientes"
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de salud del servicio (sin prefijo para fácil acceso)
    """
    return {
        "status": "healthy",
        "service": "Customer Segmentation ML",
        "timestamp": datetime.now(),
        "model_loaded": ml_service.kmeans_model is not None,
        "database_url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "configured"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Manejador global de excepciones
    """
    logger.error(f"Error no controlado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Ha ocurrido un error interno del servidor"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )