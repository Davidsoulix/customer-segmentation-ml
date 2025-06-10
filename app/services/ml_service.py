import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json

from app.models.schemas import CustomerMetrics, CustomerSegment, ClusterSummary
from app.utils.database import db_service
from app.config import settings

logger = logging.getLogger(__name__)

class MLSegmentationService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.pca_model = None
        self.feature_names = []
        self.cluster_labels = {
            0: "Champions",
            1: "Loyal Customers", 
            2: "Potential Loyalists",
            3: "New Customers",
            4: "At Risk",
            5: "Cannot Lose Them",
            6: "Hibernating",
            7: "Price Sensitive",
            8: "Promising",
            9: "Need Attention"
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara las características para el modelo de clustering
        """
        try:
            # Características principales para segmentación RFM mejorada
            features_df = pd.DataFrame()
            
            # Recency - Días desde última compra (invertido para que más reciente = mayor valor)
            features_df['recency_score'] = 1 / (df['dias_desde_ultima_compra'] + 1)
            
            # Frequency - Número de compras
            features_df['frequency'] = df['numero_compras']
            
            # Monetary - Total gastado
            features_df['monetary'] = df['total_compras']
            
            # Ticket promedio
            features_df['avg_order_value'] = df['ticket_promedio']
            
            # Diversidad de productos
            features_df['product_diversity'] = df['productos_unicos']
            
            # Frecuencia de compras (días entre compras)
            features_df['purchase_frequency'] = df['frecuencia_compras']
            
            # Características adicionales derivadas
            features_df['monetary_per_purchase'] = features_df['monetary'] / features_df['frequency']
            features_df['products_per_purchase'] = features_df['product_diversity'] / features_df['frequency']
            
            # Reemplazar infinitos y NaN
            features_df = features_df.replace([np.inf, -np.inf], 0)
            features_df = features_df.fillna(0)
            
            self.feature_names = features_df.columns.tolist()
            logger.info(f"Características preparadas: {self.feature_names}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparando características: {e}")
            raise
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_k: int = 10) -> int:
        """
        Encuentra el número óptimo de clusters usando método del codo y silhouette score
        """
        try:
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(max_k + 1, len(X) // 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=settings.RANDOM_STATE, n_init=10)
                clusters = kmeans.fit_predict(X)
                
                inertias.append(kmeans.inertia_)
                sil_score = silhouette_score(X, clusters)
                silhouette_scores.append(sil_score)
            
            # Encontrar el codo usando diferencias
            if len(inertias) >= 3:
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow_idx = np.argmax(diffs2) + 2  # +2 porque empezamos en k=2
                optimal_k_elbow = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
            else:
                optimal_k_elbow = k_range[0]
            
            # Encontrar mejor silhouette score
            best_sil_idx = np.argmax(silhouette_scores)
            optimal_k_silhouette = k_range[best_sil_idx]
            
            # Usar el promedio o el que tenga mejor silhouette score
            optimal_k = optimal_k_silhouette if silhouette_scores[best_sil_idx] > 0.5 else optimal_k_elbow
            
            logger.info(f"Clusters óptimos - Codo: {optimal_k_elbow}, Silhouette: {optimal_k_silhouette}, Elegido: {optimal_k}")
            
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error encontrando clusters óptimos: {e}")
            return 4  # Default fallback
    
    def train_model(self, usuario_ids: Optional[List[int]] = None, n_clusters: Optional[int] = None) -> Dict:
        """
        Entrena el modelo de segmentación K-means
        """
        try:
            # Obtener datos de clientes
            df = db_service.get_customer_data(usuario_ids)
            
            if df.empty:
                raise ValueError("No se encontraron datos de clientes")
            
            logger.info(f"Entrenando modelo con {len(df)} clientes")
            
            # Preparar características
            features_df = self.prepare_features(df)
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(features_df)
            
            # Determinar número óptimo de clusters
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(features_df, settings.MAX_CLUSTERS)
            
            # Entrenar modelo K-means
            self.kmeans_model = KMeans(
                n_clusters=n_clusters, 
                random_state=settings.RANDOM_STATE,
                n_init=10,
                max_iter=300
            )
            
            clusters = self.kmeans_model.fit_predict(X_scaled)
            
            # Calcular métricas de evaluación
            silhouette_avg = silhouette_score(X_scaled, clusters)
            inertia = self.kmeans_model.inertia_
            
            # Entrenar PCA para visualización
            self.pca_model = PCA(n_components=2, random_state=settings.RANDOM_STATE)
            X_pca = self.pca_model.fit_transform(X_scaled)
            
            # Asignar nombres a clusters basado en características
            cluster_centers_original = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
            cluster_names = self._assign_cluster_names(cluster_centers_original, features_df.columns)
            
            # Crear segmentos de clientes
            segments = []
            for idx, row in df.iterrows():
                segment = CustomerSegment(
                    usuario_id=int(row['usuario_id']),
                    cluster=int(clusters[idx]),
                    cluster_name=cluster_names[clusters[idx]],
                    probability=1.0  # K-means asignación hard
                )
                segments.append(segment)
            
            # Guardar modelo
            self._save_model()
            
            result = {
                'segments': segments,
                'cluster_centers': cluster_centers_original.tolist(),
                'cluster_names': cluster_names,
                'silhouette_score': float(silhouette_avg),
                'inertia': float(inertia),
                'n_clusters': n_clusters,
                'feature_names': self.feature_names,
                'pca_components': X_pca.tolist()
            }
            
            logger.info(f"Modelo entrenado exitosamente - Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            raise
    
    def _assign_cluster_names(self, cluster_centers: np.ndarray, feature_names: List[str]) -> Dict[int, str]:
        """
        Asigna nombres descriptivos a los clusters basado en sus características
        """
        cluster_names = {}
        
        # Convertir a DataFrame para análisis
        centers_df = pd.DataFrame(cluster_centers, columns=feature_names)
        
        for i, center in centers_df.iterrows():
            # Análisis RFM para nombrar clusters
            recency = center['recency_score']
            frequency = center['frequency']  
            monetary = center['monetary']
            
            # Lógica de nombrado basada en características RFM
            if monetary > centers_df['monetary'].quantile(0.75) and frequency > centers_df['frequency'].quantile(0.75):
                if recency > centers_df['recency_score'].quantile(0.75):
                    cluster_names[i] = "Champions"
                else:
                    cluster_names[i] = "Cannot Lose Them"
            elif frequency > centers_df['frequency'].quantile(0.75):
                cluster_names[i] = "Loyal Customers"
            elif monetary > centers_df['monetary'].quantile(0.75):
                if recency > centers_df['recency_score'].quantile(0.5):
                    cluster_names[i] = "Potential Loyalists"
                else:
                    cluster_names[i] = "At Risk"
            elif recency > centers_df['recency_score'].quantile(0.75):
                cluster_names[i] = "New Customers"
            elif recency < centers_df['recency_score'].quantile(0.25):
                cluster_names[i] = "Hibernating"
            else:
                cluster_names[i] = "Need Attention"
        
        return cluster_names
    
    def predict_segments(self, usuario_ids: List[int]) -> List[CustomerSegment]:
        """
        Predice segmentos para nuevos clientes
        """
        try:
            if self.kmeans_model is None:
                raise ValueError("Modelo no entrenado. Ejecute train_model() primero.")
            
            # Obtener datos de los clientes específicos
            df = db_service.get_customer_data(usuario_ids)
            
            if df.empty:
                return []
            
            # Preparar características
            features_df = self.prepare_features(df)
            
            # Escalar características
            X_scaled = self.scaler.transform(features_df)
            
            # Predecir clusters
            clusters = self.kmeans_model.predict(X_scaled)
            
            # Crear segmentos
            segments = []
            for idx, row in df.iterrows():
                segment = CustomerSegment(
                    usuario_id=int(row['usuario_id']),
                    cluster=int(clusters[idx]),
                    cluster_name=self.cluster_labels.get(clusters[idx], f"Cluster {clusters[idx]}"),
                    probability=1.0
                )
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error prediciendo segmentos: {e}")
            raise
    
    def get_cluster_summaries(self) -> List[ClusterSummary]:
        """
        Genera resúmenes de cada cluster
        """
        try:
            if self.kmeans_model is None:
                raise ValueError("Modelo no entrenado")
            
            # Obtener todos los datos de clientes
            df = db_service.get_customer_data()
            features_df = self.prepare_features(df)
            X_scaled = self.scaler.transform(features_df)
            clusters = self.kmeans_model.predict(X_scaled)
            
            summaries = []
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data = df[cluster_mask]
                
                if len(cluster_data) > 0:
                    summary = ClusterSummary(
                        cluster_id=cluster_id,
                        cluster_name=self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                        customer_count=len(cluster_data),
                        avg_total_compras=float(cluster_data['total_compras'].mean()),
                        avg_ticket_promedio=float(cluster_data['ticket_promedio'].mean()),
                        avg_frecuencia_compras=float(cluster_data['frecuencia_compras'].mean()),
                        description=f"Segmento con {len(cluster_data)} clientes"
                    )
                    summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error generando resúmenes de clusters: {e}")
            raise
    
    def _save_model(self):
        """
        Guarda el modelo entrenado
        """
        try:
            model_dir = settings.MODEL_PATH
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar componentes del modelo
            joblib.dump(self.kmeans_model, f"{model_dir}/kmeans_model_{timestamp}.pkl")
            joblib.dump(self.scaler, f"{model_dir}/scaler_{timestamp}.pkl")
            if self.pca_model:
                joblib.dump(self.pca_model, f"{model_dir}/pca_model_{timestamp}.pkl")
            
            # Guardar metadatos
            metadata = {
                'timestamp': timestamp,
                'feature_names': self.feature_names,
                'n_clusters': self.kmeans_model.n_clusters if self.kmeans_model else None,
                'cluster_labels': self.cluster_labels
            }
            
            with open(f"{model_dir}/metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modelo guardado con timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise
    
    def load_model(self, timestamp: str = None):
        """
        Carga un modelo previamente entrenado
        """
        try:
            model_dir = settings.MODEL_PATH
            
            if timestamp is None:
                # Buscar el modelo más reciente
                files = [f for f in os.listdir(model_dir) if f.startswith('kmeans_model_')]
                if not files:
                    raise ValueError("No se encontraron modelos guardados")
                files.sort(reverse=True)
                timestamp = files[0].split('_')[2].split('.')[0] + '_' + files[0].split('_')[3].split('.')[0]
            
            # Cargar componentes
            self.kmeans_model = joblib.load(f"{model_dir}/kmeans_model_{timestamp}.pkl")
            self.scaler = joblib.load(f"{model_dir}/scaler_{timestamp}.pkl")
            
            if os.path.exists(f"{model_dir}/pca_model_{timestamp}.pkl"):
                self.pca_model = joblib.load(f"{model_dir}/pca_model_{timestamp}.pkl")
            
            # Cargar metadatos
            with open(f"{model_dir}/metadata_{timestamp}.json", 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                self.cluster_labels = metadata.get('cluster_labels', self.cluster_labels)
            
            logger.info(f"Modelo cargado: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

# Instancia global del servicio ML
ml_service = MLSegmentationService()