import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Crear engine de base de datos
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DatabaseService:
    def __init__(self):
        self.engine = engine
    
    def get_customer_data(self, usuario_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Obtiene datos de clientes con sus métricas de compra
        Adaptado para tu esquema: usuario, venta, detalle_venta, producto
        """
        query = """
        WITH customer_metrics AS (
            SELECT 
                u.id as usuario_id,
                u.nombre,
                u.correo,
                u.telefono,
                u.direccion,
                u.ci,
                u.genero,
                COUNT(DISTINCT v.id) as numero_compras,
                COALESCE(SUM(v.venta_total), 0) as total_compras,
                COALESCE(AVG(v.venta_total), 0) as ticket_promedio,
                COUNT(DISTINCT dv.producto_id) as productos_unicos,
                MAX(v.fecha_venta) as ultima_compra,
                COALESCE(CURRENT_DATE - MAX(v.fecha_venta), 0) as dias_desde_ultima_compra,
                CASE 
                    WHEN COUNT(DISTINCT v.id) > 1 THEN
                        COALESCE((MAX(v.fecha_venta) - MIN(v.fecha_venta)) / 
                        NULLIF((COUNT(DISTINCT v.id) - 1), 0), 0)
                    ELSE 0 
                END as frecuencia_compras
            FROM usuario u
            LEFT JOIN venta v ON u.id = v.usuario_id AND v.estado = 'completado'
            LEFT JOIN detalle_venta dv ON v.id = dv.venta_id
            WHERE 1=1
        """
        
        params = None
        if usuario_ids:
            placeholders = ','.join(['%s'] * len(usuario_ids))
            query += f" AND u.id IN ({placeholders})"
            params = tuple(usuario_ids)  # Convertir a tupla
            
        query += """
            GROUP BY u.id, u.nombre, u.correo, u.telefono, u.direccion, u.ci, u.genero
            HAVING COUNT(DISTINCT v.id) > 0
        )
        SELECT 
            usuario_id,
            nombre,
            correo,
            telefono,
            direccion,
            ci,
            genero,
            numero_compras,
            total_compras,
            ticket_promedio,
            productos_unicos,
            COALESCE(dias_desde_ultima_compra, 0) as dias_desde_ultima_compra,
            COALESCE(frecuencia_compras, 0) as frecuencia_compras
        FROM customer_metrics
        WHERE total_compras > 0
        ORDER BY total_compras DESC
        """
        
        try:
            if params:
                df = pd.read_sql_query(query, self.engine, params=params)
            else:
                df = pd.read_sql_query(query, self.engine)
            
            # Asegurar tipos de datos correctos
            df['usuario_id'] = df['usuario_id'].astype(int)
            df['numero_compras'] = df['numero_compras'].astype(int)
            df['total_compras'] = df['total_compras'].astype(float)
            df['ticket_promedio'] = df['ticket_promedio'].astype(float)
            df['productos_unicos'] = df['productos_unicos'].astype(int)
            df['dias_desde_ultima_compra'] = df['dias_desde_ultima_compra'].astype(int)
            df['frecuencia_compras'] = df['frecuencia_compras'].astype(float)
            
            logger.info(f"Datos obtenidos: {len(df)} clientes")
            return df
        except Exception as e:
            logger.error(f"Error obteniendo datos de clientes: {e}")
            raise
    
    def get_customer_purchase_details(self, usuario_id: int) -> pd.DataFrame:
        """
        Obtiene detalles de compras de un cliente específico
        Adaptado para tu esquema con nombres de columnas correctos
        """
        query = """
        SELECT 
            v.id as venta_id,
            v.fecha_venta,
            v.venta_total,
            v.estado as venta_estado,
            v.canal_venta,
            dv.cantidad,
            dv.p_unitario as precio_unitario,
            p.nombre as producto_nombre,
            p.descripcion as producto_descripcion,
            p.p_venta as precio_venta_producto,
            p.stock,
            t.nombre as tipo_producto,
            t.descripcion as tipo_descripcion
        FROM venta v
        JOIN detalle_venta dv ON v.id = dv.venta_id
        JOIN producto p ON dv.producto_id = p.id
        LEFT JOIN tipo t ON p.tipo_id = t.id
        WHERE v.usuario_id = %s
        ORDER BY v.fecha_venta DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.engine, params=(usuario_id,))
            return df
        except Exception as e:
            logger.error(f"Error obteniendo detalles de compras: {e}")
            raise
    
    def get_product_categories(self) -> pd.DataFrame:
        """
        Obtiene información de categorías de productos
        Adaptado para tu esquema
        """
        query = """
        SELECT 
            t.id as tipo_id,
            t.nombre as tipo_nombre,
            t.descripcion as tipo_descripcion,
            COUNT(p.id) as productos_count,
            COALESCE(AVG(p.p_venta), 0) as precio_promedio
        FROM tipo t
        LEFT JOIN producto p ON t.id = p.tipo_id
        GROUP BY t.id, t.nombre, t.descripcion
        ORDER BY productos_count DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo categorías de productos: {e}")
            raise
    
    def get_sales_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de ventas por canal y estado
        """
        query = """
        SELECT 
            canal_venta,
            estado,
            COUNT(*) as total_ventas,
            SUM(venta_total) as total_ingresos,
            AVG(venta_total) as ticket_promedio,
            MIN(fecha_venta) as primera_venta,
            MAX(fecha_venta) as ultima_venta
        FROM venta
        GROUP BY canal_venta, estado
        ORDER BY total_ingresos DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo resumen de ventas: {e}")
            raise
    
    def get_customer_by_ci(self, ci: str) -> pd.DataFrame:
        """
        Obtiene datos de un cliente por su CI - CORREGIDO
        """
        query = """
        SELECT 
            id as usuario_id,
            ci,
            nombre,
            correo,
            telefono,
            direccion,
            genero
        FROM usuario
        WHERE ci = %s
        """
        
        try:
            # Usar tupla para el parámetro
            df = pd.read_sql_query(query, self.engine, params=(ci,))
            return df
        except Exception as e:
            logger.error(f"Error obteniendo cliente por CI: {e}")
            raise

# Instancia global del servicio de base de datos
db_service = DatabaseService()