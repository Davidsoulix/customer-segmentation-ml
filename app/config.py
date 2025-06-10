from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database - Adaptado para tu configuración
    DATABASE_URL: str = "postgresql://postgres:2405@localhost:5432/machine"
    
    # ML Model settings
    MODEL_PATH: str = "models/"
    MIN_CLUSTERS: int = 2
    MAX_CLUSTERS: int = 10
    RANDOM_STATE: int = 42
    
    # API settings
    API_TITLE: str = "Customer Segmentation ML Service"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Configuraciones específicas para tu esquema
    # Canales de venta disponibles
    SALES_CHANNELS: list = ["online", "tienda", "telefono", "app"]
    
    # Estados de venta disponibles
    SALES_STATES: list = ["completada", "pendiente", "cancelada", "reembolsada"]
    
    # Géneros disponibles
    GENDERS: list = ["masculino", "femenino", "otro", "no_especificado"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()