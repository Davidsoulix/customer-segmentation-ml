#!/usr/bin/env python3
"""
Script de prueba completo para el Microservicio de Machine Learning
Customer Segmentation Service
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional

class MLServiceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.session = requests.Session()
        self.job_id = None
        
    def print_header(self, title: str):
        """Imprime un header decorativo"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_step(self, step: str):
        """Imprime un paso de prueba"""
        print(f"\n🔍 {step}")
        print("-" * 50)
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Hace una petición HTTP con manejo de errores"""
        url = f"{self.api_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            print(f"   📡 {method} {endpoint}")
            print(f"   📊 Status: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error en petición: {e}")
            return None
    
    def test_service_health(self):
        """Prueba el estado del servicio"""
        self.print_step("1. Verificando estado del servicio")
        
        # Root endpoint
        response = requests.get(f"{self.base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Servicio activo: {data['service']} v{data['version']}")
        else:
            print(f"   ❌ Servicio no disponible: {response.status_code}")
            return False
        
        # Health check
        response = requests.get(f"{self.base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check: {data['status']}")
            print(f"   📊 Modelo cargado: {data['model_loaded']}")
        else:
            print(f"   ❌ Health check falló: {response.status_code}")
            return False
            
        return True
    
    def test_customer_data(self):
        """Prueba la obtención de datos de clientes"""
        self.print_step("2. Obteniendo datos de clientes")
        
        response = self.make_request("GET", "/customers/data")
        if response and response.status_code == 200:
            customers = response.json()
            print(f"   ✅ Clientes obtenidos: {len(customers)}")
            
            if customers:
                sample_customer = customers[0]
                print(f"   👤 Cliente ejemplo: {sample_customer['nombre']}")
                print(f"   💰 Total compras: ${sample_customer['total_compras']:.2f}")
                print(f"   🛒 Número compras: {sample_customer['numero_compras']}")
                print(f"   📅 Días desde última compra: {sample_customer['dias_desde_ultima_compra']}")
                return customers
        else:
            print(f"   ❌ Error obteniendo datos de clientes")
            return []
    
    def test_customer_by_ci(self):
        """Prueba la búsqueda de cliente por CI"""
        self.print_step("3. Buscando cliente por CI")
        
        # Usar un CI conocido de los datos de ejemplo
        ci = "12345678"  # Juan Pérez
        response = self.make_request("GET", f"/customers/by-ci/{ci}")
        
        if response and response.status_code == 200:
            customers = response.json()
            if customers:
                customer = customers[0]
                print(f"   ✅ Cliente encontrado: {customer['nombre']}")
                print(f"   📧 Email: {customer['correo']}")
                print(f"   📱 Teléfono: {customer.get('telefono', 'N/A')}")
                return customer['usuario_id']
        else:
            print(f"   ❌ Cliente no encontrado o error en búsqueda")
            return None
    
    def test_customer_purchases(self, usuario_id: int):
        """Prueba la obtención de compras de un cliente"""
        self.print_step(f"4. Obteniendo compras del cliente {usuario_id}")
        
        response = self.make_request("GET", f"/customers/{usuario_id}/purchases")
        
        if response and response.status_code == 200:
            purchases = response.json()
            print(f"   ✅ Compras obtenidas: {len(purchases)}")
            
            if purchases:
                for i, purchase in enumerate(purchases[:3]):  # Mostrar solo las primeras 3
                    print(f"   🛍️  Compra {i+1}: ${purchase['venta_total']:.2f} - {purchase['producto_nombre']}")
                    
                total_spent = sum(p['venta_total'] for p in purchases)
                print(f"   💰 Total gastado: ${total_spent:.2f}")
            return purchases
        else:
            print(f"   ❌ Error obteniendo compras del cliente")
            return []
    
    def test_sales_summary(self):
        """Prueba el resumen de ventas"""
        self.print_step("5. Obteniendo resumen de ventas")
        
        response = self.make_request("GET", "/sales/summary")
        
        if response and response.status_code == 200:
            summaries = response.json()
            print(f"   ✅ Resúmenes obtenidos: {len(summaries)}")
            
            for summary in summaries[:5]:  # Mostrar primeras 5
                print(f"   📊 {summary['canal_venta']} - {summary['estado']}: "
                      f"{summary['total_ventas']} ventas, ${summary['total_ingresos']:.2f}")
            return summaries
        else:
            print(f"   ❌ Error obteniendo resumen de ventas")
            return []
    
    def test_product_categories(self):
        """Prueba la obtención de categorías de productos"""
        self.print_step("6. Obteniendo categorías de productos")
        
        response = self.make_request("GET", "/products/categories")
        
        if response and response.status_code == 200:
            categories = response.json()
            print(f"   ✅ Categorías obtenidas: {len(categories)}")
            
            for category in categories[:5]:  # Mostrar primeras 5
                print(f"   🏷️  {category['tipo_nombre']}: {category['productos_count']} productos")
            return categories
        else:
            print(f"   ❌ Error obteniendo categorías de productos")
            return []
    
    def test_model_training(self):
        """Prueba el entrenamiento del modelo"""
        self.print_step("7. Entrenando modelo de segmentación")
        
        # Crear solicitud de entrenamiento
        training_request = {
            "n_clusters": 4,  # Usar 4 clusters
            "usuario_ids": None  # Entrenar con todos los datos
        }
        
        response = self.make_request("POST", "/segment/train", json=training_request)
        
        if response and response.status_code == 200:
            result = response.json()
            self.job_id = result['job_id']
            print(f"   ✅ Entrenamiento iniciado")
            print(f"   🆔 Job ID: {self.job_id}")
            print(f"   ⏳ Estado: {result['status']}")
            return self.job_id
        else:
            print(f"   ❌ Error iniciando entrenamiento")
            return None
    
    def test_job_status(self, job_id: str, max_wait: int = 60):
        """Prueba el estado del job y espera a que complete"""
        self.print_step(f"8. Monitoreando job {job_id}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = self.make_request("GET", f"/segment/status/{job_id}")
            
            if response and response.status_code == 200:
                job_status = response.json()
                status = job_status['status']
                print(f"   ⏱️  Estado actual: {status}")
                
                if status == "completed":
                    print(f"   ✅ Job completado exitosamente")
                    completion_time = job_status.get('completed_at')
                    if completion_time:
                        print(f"   🕐 Completado en: {completion_time}")
                    return True
                elif status == "failed":
                    error = job_status.get('error_message', 'Error desconocido')
                    print(f"   ❌ Job falló: {error}")
                    return False
                elif status in ["pending", "processing"]:
                    print(f"   ⏳ Esperando... ({int(time.time() - start_time)}s)")
                    time.sleep(5)
                else:
                    print(f"   ⚠️  Estado desconocido: {status}")
                    time.sleep(2)
            else:
                print(f"   ❌ Error obteniendo estado del job")
                return False
        
        print(f"   ⏰ Timeout alcanzado ({max_wait}s)")
        return False
    
    def test_segmentation_results(self, job_id: str):
        """Prueba la obtención de resultados de segmentación"""
        self.print_step(f"9. Obteniendo resultados de segmentación")
        
        response = self.make_request("GET", f"/segment/results/{job_id}")
        
        if response and response.status_code == 200:
            results = response.json()
            segments = results['segments']
            
            print(f"   ✅ Segmentación completada")
            print(f"   👥 Clientes segmentados: {len(segments)}")
            print(f"   📊 Silhouette Score: {results['silhouette_score']:.3f}")
            print(f"   📉 Inertia: {results['inertia']:.2f}")
            
            # Mostrar distribución de clusters
            cluster_counts = {}
            for segment in segments:
                cluster_name = segment['cluster_name']
                cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
            
            print(f"   🏷️  Distribución por clusters:")
            for cluster_name, count in cluster_counts.items():
                print(f"      • {cluster_name}: {count} clientes")
                
            return results
        else:
            print(f"   ❌ Error obteniendo resultados de segmentación")
            return None
    
    def test_cluster_summaries(self):
        """Prueba la obtención de resúmenes de clusters"""
        self.print_step("10. Obteniendo resúmenes de clusters")
        
        response = self.make_request("GET", "/segment/clusters/summary")
        
        if response and response.status_code == 200:
            summaries = response.json()
            print(f"   ✅ Resúmenes obtenidos: {len(summaries)}")
            
            for summary in summaries:
                print(f"   🏷️  {summary['cluster_name']}:")
                print(f"      👥 Clientes: {summary['customer_count']}")
                print(f"      💰 Compras promedio: ${summary['avg_total_compras']:.2f}")
                print(f"      🎫 Ticket promedio: ${summary['avg_ticket_promedio']:.2f}")
                print(f"      📅 Frecuencia promedio: {summary['avg_frecuencia_compras']:.1f} días")
                print()
            
            return summaries
        else:
            print(f"   ❌ Error obteniendo resúmenes de clusters")
            return []
    
    def test_individual_prediction(self, usuario_ids: List[int]):
        """Prueba la predicción individual de segmentos"""
        self.print_step("11. Prediciendo segmentos individuales")
        
        response = self.make_request("POST", "/segment/predict", json=usuario_ids[:3])  # Probar con primeros 3
        
        if response and response.status_code == 200:
            predictions = response.json()
            print(f"   ✅ Predicciones obtenidas: {len(predictions)}")
            
            for pred in predictions:
                print(f"   👤 Usuario {pred['usuario_id']}: {pred['cluster_name']} "
                      f"(Cluster {pred['cluster']}, Probabilidad: {pred['probability']:.2f})")
            
            return predictions
        else:
            print(f"   ❌ Error prediciendo segmentos individuales")
            return []
    
    def test_customer_segment(self, usuario_id: int):
        """Prueba la obtención del segmento de un cliente específico"""
        self.print_step(f"12. Obteniendo segmento del cliente {usuario_id}")
        
        response = self.make_request("GET", f"/customers/{usuario_id}/segment")
        
        if response and response.status_code == 200:
            segment = response.json()
            print(f"   ✅ Segmento obtenido")
            print(f"   👤 Usuario: {segment['usuario_id']}")
            print(f"   🏷️  Cluster: {segment['cluster_name']} (ID: {segment['cluster']})")
            print(f"   📊 Probabilidad: {segment['probability']:.2f}")
            
            return segment
        else:
            print(f"   ❌ Error obteniendo segmento del cliente")
            return None
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        self.print_header("INICIO DE PRUEBAS DEL MICROSERVICIO ML")
        
        print(f"🚀 Iniciando pruebas del microservicio en: {self.base_url}")
        print(f"🕐 Tiempo de inicio: {datetime.now()}")
        
        try:
            # 1. Verificar estado del servicio
            if not self.test_service_health():
                print("\n❌ Servicio no disponible. Deteniendo pruebas.")
                return False
            
            # 2. Obtener datos de clientes
            customers = self.test_customer_data()
            if not customers:
                print("\n❌ No se pudieron obtener datos de clientes.")
                return False
            
            # 3. Buscar cliente por CI
            usuario_id = self.test_customer_by_ci()
            if not usuario_id:
                usuario_id = customers[0]['usuario_id']  # Usar el primero como fallback
            
            # 4. Obtener compras del cliente
            self.test_customer_purchases(usuario_id)
            
            # 5. Obtener resumen de ventas
            self.test_sales_summary()
            
            # 6. Obtener categorías de productos
            self.test_product_categories()
            
            # 7. Entrenar modelo
            job_id = self.test_model_training()
            if not job_id:
                print("\n❌ No se pudo iniciar el entrenamiento.")
                return False
            
            # 8. Monitorear job
            if not self.test_job_status(job_id, max_wait=120):  # 2 minutos máximo
                print("\n❌ El job no completó en el tiempo esperado.")
                return False
            
            # 9. Obtener resultados
            results = self.test_segmentation_results(job_id)
            if not results:
                print("\n❌ No se pudieron obtener los resultados.")
                return False
            
            # 10. Obtener resúmenes de clusters
            self.test_cluster_summaries()
            
            # 11. Predicciones individuales
            usuario_ids = [c['usuario_id'] for c in customers[:5]]
            self.test_individual_prediction(usuario_ids)
            
            # 12. Segmento de cliente específico
            self.test_customer_segment(usuario_id)
            
            self.print_header("TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️  Pruebas interrumpidas por el usuario")
            return False
        except Exception as e:
            print(f"\n❌ Error inesperado durante las pruebas: {e}")
            return False
        finally:
            print(f"\n🕐 Tiempo de finalización: {datetime.now()}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Script de prueba para Microservicio ML")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="URL base del microservicio (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test",
        choices=['all', 'health', 'data', 'training', 'prediction'],
        default='all',
        help="Tipo de prueba a ejecutar (default: all)"
    )
    
    args = parser.parse_args()
    
    tester = MLServiceTester(args.url)
    
    if args.test == 'all':
        success = tester.run_all_tests()
    elif args.test == 'health':
        success = tester.test_service_health()
    elif args.test == 'data':
        tester.test_customer_data()
        success = True
    elif args.test == 'training':
        job_id = tester.test_model_training()
        if job_id:
            success = tester.test_job_status(job_id)
        else:
            success = False
    elif args.test == 'prediction':
        # Primero obtener algunos IDs de usuario
        customers = tester.test_customer_data()
        if customers:
            usuario_ids = [c['usuario_id'] for c in customers[:3]]
            tester.test_individual_prediction(usuario_ids)
            success = True
        else:
            success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()