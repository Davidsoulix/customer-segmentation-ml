from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum

class SegmentationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CustomerData(BaseModel):
    usuario_id: int
    ci: str
    nombre: str
    correo: str 
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    genero: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True

class CustomerMetrics(BaseModel):
    usuario_id: int
    ci: Optional[str] = None
    nombre: str
    correo: str = Field(description="Correo electrónico del cliente")
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    genero: Optional[str] = None
    total_compras: float = Field(description="Total de dinero gastado por el cliente")
    numero_compras: int = Field(description="Número total de compras")
    frecuencia_compras: float = Field(description="Días promedio entre compras")
    ticket_promedio: float = Field(description="Valor promedio por compra")
    dias_desde_ultima_compra: int = Field(description="Días desde la última compra")
    productos_unicos: int = Field(description="Número de productos únicos comprados")
    
class CustomerSegment(BaseModel):
    usuario_id: int
    cluster: int
    cluster_name: str
    probability: float = Field(description="Probabilidad de pertenencia al cluster", ge=0, le=1)
    
class SegmentationRequest(BaseModel):
    n_clusters: Optional[int] = Field(default=None, ge=2, le=10)
    features: Optional[List[str]] = Field(default=None, description="Features específicas a usar")
    usuario_ids: Optional[List[int]] = Field(default=None, description="IDs específicos de usuarios")
    
class SegmentationResponse(BaseModel):
    job_id: str
    status: SegmentationStatus
    segments: List[CustomerSegment]
    #cluster_centers: Dict[str, Any]
    cluster_centers: List[List[float]]  # <- Cambio aquí
    silhouette_score: float
    inertia: float
    created_at: datetime
    
class ClusterSummary(BaseModel):
    cluster_id: int
    cluster_name: str
    customer_count: int
    avg_total_compras: float
    avg_ticket_promedio: float
    avg_frecuencia_compras: float
    description: str
    
class SegmentationJob(BaseModel):
    job_id: str
    status: SegmentationStatus
    parameters: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None  # ✅ AÑADIDO: Campo para almacenar resultados

# Modelos adicionales para tu esquema específico
class ProductoData(BaseModel):
    id: int
    nombre: str
    p_venta: float = Field(description="Precio de venta")
    imagen: Optional[str] = None
    stock: int
    descripcion: Optional[str] = None
    tipo_id: int

class TipoData(BaseModel):
    id: int
    nombre: str
    descripcion: Optional[str] = None

class VentaData(BaseModel):
    id: int
    fecha_venta: date
    venta_total: float
    estado: str
    canal_venta: str
    usuario_id: int

class DetalleVentaData(BaseModel):
    id: int
    cantidad: int
    p_unitario: float = Field(description="Precio unitario")
    producto_id: int
    venta_id: int

class CustomerPurchaseDetail(BaseModel):
    venta_id: int
    fecha_venta: date
    venta_total: float
    venta_estado: str
    canal_venta: str
    cantidad: int
    precio_unitario: float
    producto_nombre: str
    producto_descripcion: Optional[str] = None
    precio_venta_producto: float
    stock: int
    tipo_producto: Optional[str] = None
    tipo_descripcion: Optional[str] = None

class SalesSummary(BaseModel):
    canal_venta: str
    estado: str
    total_ventas: int
    total_ingresos: float
    ticket_promedio: float
    primera_venta: date
    ultima_venta: date