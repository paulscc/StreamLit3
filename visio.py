
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import requests
import os
import numpy as np
import re
import streamlit as st
import cv2
# Intentar cargar modelo YOLO
try:
    model_barcode = YOLO("YOLOV8s_Barcode_Detection.pt")
    print("Modelo YOLO personalizado cargado")
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"No se pudo cargar modelo personalizado: {e}")
    try:
        model_barcode = YOLO("yolov8n.pt")
        print("Modelo YOLO general cargado")
        YOLO_AVAILABLE = True
    except Exception as e2:
        print(f"No se pudo cargar ningún modelo YOLO: {e2}")
        model_barcode = None
        YOLO_AVAILABLE = False

def buscar_libro_openlibrary(isbn):
    """Buscar información del libro en OpenLibrary"""
    try:
        # Limpiar ISBN
        isbn_clean = re.sub(r'[^0-9X]', '', str(isbn).upper())
        
        if len(isbn_clean) < 10:
            print(f"ISBN muy corto: {isbn_clean}")
            return None
            
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn_clean}&format=json&jscmd=data"
        print(f"Buscando libro con ISBN: {isbn_clean}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            key = f"ISBN:{isbn_clean}"
            
            if key in data and data[key]:
                libro = data[key]
                return {
                    "titulo": libro.get("title", "Sin título"),
                    "autores": ", ".join([a.get("name", "") for a in libro.get("authors", [])]) or "Sin autor",
                    "editorial": ", ".join([p.get("name", "") for p in libro.get("publishers", [])]) or "Sin editorial",
                    "isbn": isbn_clean
                }
            else:
                print(f"No encontrado en OpenLibrary: {isbn_clean}")
        else:
            print(f"Error HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error buscando libro: {e}")
    
    return None

class VideoProcessor:
    def __init__(self):
        self.detected_barcodes = set()
        self.frame_count = 0
        
        print(f"YOLO disponible: {YOLO_AVAILABLE}")
        if not model_barcode:
            print("Advertencia: No se pudo cargar modelo YOLO")
    
    def detect_with_pyzbar(self, image):
        """Detectar códigos de barras usando pyzbar"""
        results = []
        
        try:
            # Convertir a escala de grises para mejor detección
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar códigos de barras
            decoded_objects = decode(gray)
            
            for obj in decoded_objects:
                # Extraer datos del código
                data = obj.data.decode('utf-8')
                barcode_type = obj.type
                
                # Obtener coordenadas del rectángulo
                rect = obj.rect
                bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                
                results.append({
                    'data': data,
                    'type': barcode_type,
                    'bbox': bbox,
                    'method': 'pyzbar'
                })
                
                print(f"pyzbar detectó: {data} tipo: {barcode_type}")
                
        except Exception as e:
            print(f"Error en pyzbar: {e}")
            
        return results
    
    def detect_with_yolo_and_pyzbar(self, image):
        """Usar YOLO para encontrar regiones y pyzbar para decodificar"""
        results = []
        
        if not YOLO_AVAILABLE or model_barcode is None:
            return results
            
        try:
            # Usar YOLO para detección
            yolo_results = model_barcode.predict(image, verbose=False, conf=0.2)
            
            for result in yolo_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Expandir ROI para mejor captura
                        margin = 20
                        h, w = image.shape[:2]
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        roi = image[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            # Usar pyzbar en la ROI
                            roi_results = self.detect_with_pyzbar(roi)
                            
                            for roi_result in roi_results:
                                # Ajustar coordenadas al frame original
                                orig_bbox = roi_result['bbox']
                                adjusted_bbox = (
                                    orig_bbox[0] + x1,
                                    orig_bbox[1] + y1,
                                    orig_bbox[2] + x1,
                                    orig_bbox[3] + y1
                                )
                                roi_result['bbox'] = adjusted_bbox
                                roi_result['method'] = 'yolo+pyzbar'
                                results.append(roi_result)
                                
        except Exception as e:
            print(f"Error en YOLO+pyzbar: {e}")
            
        return results
    
    def process_frame_for_webrtc(self, frame):
        """Procesar frame principal"""
        self.frame_count += 1
        info_list = []
        all_detections = []
        
        # Debug cada 30 frames
        if self.frame_count % 30 == 0:
            print(f"Frame {self.frame_count}: Procesando...")
        
        # Método 1: YOLO + pyzbar en ROI (más preciso)
        yolo_detections = self.detect_with_yolo_and_pyzbar(frame)
        all_detections.extend(yolo_detections)
        
        # Método 2: pyzbar directo en toda la imagen (fallback)
        if not all_detections:
            direct_detections = self.detect_with_pyzbar(frame)
            all_detections.extend(direct_detections)
        
        # Método 3: Crear detección de prueba cada 120 frames
        if not all_detections and self.frame_count % 120 == 0:
            test_detection = [{
                'data': '9780134685991',  # ISBN de prueba válido
                'type': 'TEST_CODE128',
                'bbox': (frame.shape[1]//4, frame.shape[0]//4, 3*frame.shape[1]//4, 3*frame.shape[0]//4),
                'method': 'test_simulation'
            }]
            all_detections.extend(test_detection)
            print("Creando detección de prueba...")
        
        # Procesar todas las detecciones
        for detection in all_detections:
            data = detection['data']
            det_type = detection['type']
            method = detection.get('method', 'unknown')
            bbox = detection.get('bbox')
            
            print(f"Detectado: {data} método: {method}")
            
            # Dibujar en el frame
            if bbox:
                x1, y1, x2, y2 = bbox
            else:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4
            
            # Color según el método
            if method == 'test_simulation':
                color = (255, 0, 255)  # Magenta para pruebas
            elif 'pyzbar' in method:
                color = (0, 255, 0)    # Verde para pyzbar
            else:
                color = (255, 255, 0)  # Cian para otros
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Dibujar texto principal
            text = f"{det_type}: {data[:15]}..."
            cv2.putText(frame, text, (x1, max(10, y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Texto del método
            cv2.putText(frame, f"Método: {method}", (x1, y1+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Procesar código nuevo
            if data not in self.detected_barcodes:
                self.detected_barcodes.add(data)
                print(f"Nuevo código: {data}")
                
                # Buscar información del libro
                if method == 'test_simulation':
                    # Para pruebas, crear info de ejemplo
                    book_info = {
                        "titulo": "Computer Science: An Overview (12th Edition)",
                        "autores": "J. Glenn Brookshear, Dennis Brylow",
                        "editorial": "Pearson",
                        "isbn": data
                    }
                else:
                    book_info = buscar_libro_openlibrary(data)
                
                info_list.append({
                    "isbn": data,
                    "barcode_type": det_type,
                    "detection_method": method,
                    "book_info": book_info,
                    "frame": self.frame_count
                })
                
                if book_info:
                    print(f"Libro encontrado: {book_info.get('titulo', 'Sin título')}")
                else:
                    print("No se encontró información del libro")
        
        # Mostrar estadísticas en el frame
        stats_text = f"Frame: {self.frame_count} | Detectados: {len(all_detections)} | Únicos: {len(self.detected_barcodes)}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return frame, info_list
    
    def clear_detected_barcodes(self):
        """Limpiar códigos detectados"""
        self.detected_barcodes.clear()
        self.frame_count = 0
        print("Códigos detectados limpiados")



