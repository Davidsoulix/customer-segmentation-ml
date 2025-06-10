from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
import uuid
from datetime import datetime
import logging
import traceback
import pandas as pd

from app.models.schemas import (
    SegmentationRequest, SegmentationResponse, CustomerSegment,
    ClusterSummary, SegmentationJob, SegmentationStatus,
    CustomerPurchaseDetail, SalesSummary, CustomerData
)
from app.services.ml_service import ml_service
from app.utils.database import db_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Storage en memoria para jobs (en producción usar Redis o base de datos)
jobs_storage = {}

@router.post("/segment/train", response_model=SegmentationResponse)
async def train_segmentation_model(
    background_tasks: BackgroundTasks,
    request: SegmentationRequest
):
    """
    Entrena el modelo de segmentación K-means
    """
    try:
        job_id = str(uuid.uuid4())
        
        # Crear job en storage
        job = SegmentationJob(
            job_id=job_id,
            status=SegmentationStatus.PROCESSING,
            parameters=request.dict(),
            created_at=datetime.now()
        )
        jobs_storage[job_id] = job
        
        # Ejecutar entrenamiento en background
        background_tasks.add_task(
            _train_model_background,
            job_id,
            request.usuario_ids,
            request.n_clusters
        )
        
        # Respuesta inicial
        return SegmentationResponse(
            job_id=job_id,
            status=SegmentationStatus.PROCESSING,
            segments=[],
            cluster_centers=[],
            silhouette_score=0.0,
            inertia=0.0,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segment/status/{job_id}", response_model=SegmentationJob)
async def get_segmentation_status(job_id: str):
    """
    Obtiene el estado de un job de segmentación
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    return jobs_storage[job_id]

@router.get("/segment/results/{job_id}", response_model=SegmentationResponse)
async def get_segmentation_results(job_id: str):
    """
    Obtiene los resultados de una segmentación completada
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    job = jobs_storage[job_id]
    
    if job.status != SegmentationStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job en estado: {job.status}. Debe estar completado."
        )
    
    # Verificar que los resultados existen
    if not job.results:
        raise HTTPException(status_code=404, detail="Resultados no encontrados")
    
    # Convertir dict a SegmentationResponse
    try:
        return SegmentationResponse(**job.results)
    except Exception as e:
        logger.error(f"Error convirtiendo resultados: {e}")
        raise HTTPException(status_code=500, detail="Error procesando resultados")

@router.post("/segment/predict", response_model=List[CustomerSegment])
async def predict_customer_segments(usuario_ids: List[int]):
    """
    Predice segmentos para clientes específicos usando modelo entrenado
    """
    try:
        segments = ml_service.predict_segments(usuario_ids)
        return segments
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error prediciendo segmentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segment/clusters/summary", response_model=List[ClusterSummary])
async def get_cluster_summaries():
    """
    Obtiene resúmenes de todos los clusters
    """
    try:
        summaries = ml_service.get_cluster_summaries()
        return summaries
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo resúmenes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/{usuario_id}/segment", response_model=CustomerSegment)
async def get_customer_segment(usuario_id: int):
    """
    Obtiene el segmento de un cliente específico
    """
    try:
        segments = ml_service.predict_segments([usuario_id])
        if not segments:
            raise HTTPException(status_code=404, detail="Cliente no encontrado")
        return segments[0]
    except Exception as e:
        logger.error(f"Error obteniendo segmento del cliente: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/data")
async def get_customers_data(usuario_ids: Optional[List[int]] = None):
    """
    Obtiene datos de clientes para análisis
    """
    try:
        df = db_service.get_customer_data(usuario_ids)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error obteniendo datos de clientes: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/{usuario_id}/purchases", response_model=List[CustomerPurchaseDetail])
async def get_customer_purchases(usuario_id: int):
    """
    Obtiene historial de compras de un cliente con detalles adaptados
    """
    try:
        df = db_service.get_customer_purchase_details(usuario_id)
        
        if df.empty:
            return []
        
        # Convertir DataFrame a lista de objetos Pydantic
        purchases = []
        for _, row in df.iterrows():
            purchase = CustomerPurchaseDetail(
                venta_id=int(row['venta_id']),
                fecha_venta=row['fecha_venta'],
                venta_total=float(row['venta_total']),
                venta_estado=str(row['venta_estado']),
                canal_venta=str(row['canal_venta']),
                cantidad=int(row['cantidad']),
                precio_unitario=float(row['precio_unitario']),
                producto_nombre=str(row['producto_nombre']),
                producto_descripcion=str(row.get('producto_descripcion', '')),
                precio_venta_producto=float(row['precio_venta_producto']),
                stock=int(row['stock']),
                tipo_producto=str(row.get('tipo_producto', '')),
                tipo_descripcion=str(row.get('tipo_descripcion', ''))
            )
            purchases.append(purchase)
        return purchases
    except Exception as e:
        logger.error(f"Error obteniendo compras del cliente: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/by-ci/{ci}", response_model=List[CustomerData])
async def get_customer_by_ci(ci: str):
    """
    Obtiene datos de un cliente por su CI
    """
    try:
        df = db_service.get_customer_by_ci(ci)
        if df.empty:
            raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
        customers = []
        for _, row in df.iterrows():
            customer = CustomerData(
                usuario_id=int(row['usuario_id']),
                ci=str(row['ci']),
                nombre=str(row['nombre']),
                correo=str(row['correo']),
                telefono=str(row.get('telefono', '')),
                direccion=str(row.get('direccion', '')),
                genero=str(row.get('genero', ''))
            )
            customers.append(customer)
        return customers
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo cliente por CI: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sales/summary", response_model=List[SalesSummary])
async def get_sales_summary():
    """
    Obtiene resumen de ventas por canal y estado
    """
    try:
        df = db_service.get_sales_summary()
        summaries = []
        for _, row in df.iterrows():
            summary = SalesSummary(
                canal_venta=str(row['canal_venta']),
                estado=str(row['estado']),
                total_ventas=int(row['total_ventas']),
                total_ingresos=float(row['total_ingresos']),
                ticket_promedio=float(row['ticket_promedio']),
                primera_venta=row['primera_venta'],
                ultima_venta=row['ultima_venta']
            )
            summaries.append(summary)
        return summaries
    except Exception as e:
        logger.error(f"Error obteniendo resumen de ventas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/categories")
async def get_product_categories():
    """
    Obtiene información de categorías de productos
    """
    try:
        df = db_service.get_product_categories()
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error obteniendo categorías de productos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Re-entrena el modelo con todos los datos disponibles
    """
    try:
        job_id = str(uuid.uuid4())
        
        job = SegmentationJob(
            job_id=job_id,
            status=SegmentationStatus.PROCESSING,
            parameters={"retrain": True},
            created_at=datetime.now()
        )
        jobs_storage[job_id] = job
        
        background_tasks.add_task(_train_model_background, job_id, None, None)
        
        return {"job_id": job_id, "message": "Re-entrenamiento iniciado"}
        
    except Exception as e:
        logger.error(f"Error iniciando re-entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Endpoint de salud del servicio
    """
    return {
        "status": "healthy",
        "service": "Customer Segmentation ML",
        "timestamp": datetime.now(),
        "model_loaded": ml_service.kmeans_model is not None
    }

async def _train_model_background(
    job_id: str, 
    usuario_ids: Optional[List[int]], 
    n_clusters: Optional[int]
):
    """
    Función background para entrenar el modelo
    """
    try:
        logger.info(f"Iniciando entrenamiento para job {job_id}")
        
        # Actualizar estado del job
        if job_id in jobs_storage:
            jobs_storage[job_id].status = SegmentationStatus.PROCESSING
        
        # Entrenar modelo
        result = ml_service.train_model(usuario_ids, n_clusters)
        
        # Preparar datos para SegmentationResponse
        response_data = {
            "job_id": job_id,
            "status": SegmentationStatus.COMPLETED,
            "segments": [segment.dict() for segment in result['segments']],  # ✅ Convertir a dict
            "cluster_centers": result['cluster_centers'],
            "silhouette_score": result['silhouette_score'],
            "inertia": result['inertia'],
            "created_at": datetime.now()
        }
        
        # Actualizar job con resultados
        if job_id in jobs_storage:
            jobs_storage[job_id].status = SegmentationStatus.COMPLETED
            jobs_storage[job_id].completed_at = datetime.now()
            jobs_storage[job_id].results = response_data  # ✅ Guardar como dict
        
        logger.info(f"Entrenamiento completado para job {job_id}")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento background para job {job_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Actualizar job con error
        if job_id in jobs_storage:
            jobs_storage[job_id].status = SegmentationStatus.FAILED
            jobs_storage[job_id].error_message = str(e)
            jobs_storage[job_id].completed_at = datetime.now()