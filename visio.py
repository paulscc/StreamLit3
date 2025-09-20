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
        self.processing_mode = 'auto'  # 'auto', 'image', 'video', 'webcam'
        
        print(f"YOLO disponible: {YOLO_AVAILABLE}")
        if not model_barcode:
            print("Advertencia: No se pudo cargar modelo YOLO")
    
    def set_processing_mode(self, mode):
        """Establecer modo de procesamiento"""
        self.processing_mode = mode
        print(f"Modo de procesamiento: {mode}")
    
    def detect_with_pyzbar(self, image):
        """Detectar códigos de barras usando pyzbar con múltiples técnicas"""
        results = []
        
        try:
            # Técnica 1: Imagen original
            results.extend(self._pyzbar_decode_image(image, "original"))
            
            # Técnica 2: Escala de grises
            if len(results) == 0:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                results.extend(self._pyzbar_decode_image(gray, "grayscale"))
            
            # Técnica 3: Mejorar contraste para imágenes estáticas
            if len(results) == 0 and self.processing_mode == 'image':
                enhanced = self._enhance_image_for_barcode(image)
                results.extend(self._pyzbar_decode_image(enhanced, "enhanced"))
            
            # Técnica 4: Diferentes escalas para imágenes
            if len(results) == 0 and self.processing_mode == 'image':
                for scale in [0.5, 1.5, 2.0]:
                    scaled = self._scale_image(image, scale)
                    if scaled is not None:
                        scaled_results = self._pyzbar_decode_image(scaled, f"scaled_{scale}")
                        if scaled_results:
                            # Ajustar coordenadas según la escala
                            for result in scaled_results:
                                bbox = result['bbox']
                                adjusted_bbox = tuple(int(coord / scale) for coord in bbox)
                                result['bbox'] = adjusted_bbox
                                result['method'] += f"_rescaled"
                            results.extend(scaled_results)
                            break
                
        except Exception as e:
            print(f"Error en pyzbar: {e}")
            
        return results
    
    def _pyzbar_decode_image(self, image, technique_name):
        """Decodificar imagen con pyzbar"""
        results = []
        try:
            decoded_objects = decode(image)
            
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                barcode_type = obj.type
                
                # Obtener coordenadas del rectángulo
                rect = obj.rect
                bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                
                results.append({
                    'data': data,
                    'type': barcode_type,
                    'bbox': bbox,
                    'method': f'pyzbar_{technique_name}'
                })
                
                print(f"pyzbar ({technique_name}) detectó: {data} tipo: {barcode_type}")
                
        except Exception as e:
            print(f"Error en pyzbar {technique_name}: {e}")
            
        return results
    
    def _enhance_image_for_barcode(self, image):
        """Mejorar imagen para detección de códigos de barras"""
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Ecualización de histograma
            equalized = cv2.equalizeHist(gray)
            
            # Filtro bilateral para reducir ruido manteniendo bordes
            bilateral = cv2.bilateralFilter(equalized, 9, 75, 75)
            
            # Sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(bilateral, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            print(f"Error mejorando imagen: {e}")
            return image
    
    def _scale_image(self, image, scale_factor):
        """Escalar imagen manteniendo aspect ratio"""
        try:
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            if new_width > 50 and new_height > 50 and new_width < 5000 and new_height < 5000:
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
        except Exception as e:
            print(f"Error escalando imagen: {e}")
            
        return None
    
    def detect_with_yolo_and_pyzbar(self, image):
        """Usar YOLO para encontrar regiones y pyzbar para decodificar"""
        results = []
        
        if not YOLO_AVAILABLE or model_barcode is None:
            return results
            
        try:
            # Configurar confianza según el modo
            confidence = 0.1 if self.processing_mode == 'image' else 0.2
            
            # Usar YOLO para detección
            yolo_results = model_barcode.predict(image, verbose=False, conf=confidence)
            
            for result in yolo_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Expandir ROI para mejor captura
                        margin = 30 if self.processing_mode == 'image' else 20
                        h, w = image.shape[:2]
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        roi = image[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            # Usar pyzbar en la ROI
                            roi_results = self._pyzbar_decode_image(roi, "yolo_roi")
                            
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
    
    def process_image_static(self, image):
        """Procesar imagen estática (optimizado para imágenes subidas)"""
        self.set_processing_mode('image')
        info_list = []
        all_detections = []
        
        print("Procesando imagen estática...")
        
        # Crear copia para dibujar
        annotated_image = image.copy()
        
        # Método 1: YOLO + pyzbar en ROI (más preciso)
        yolo_detections = self.detect_with_yolo_and_pyzbar(image)
        all_detections.extend(yolo_detections)
        
        # Método 2: pyzbar directo con múltiples técnicas
        if not all_detections:
            direct_detections = self.detect_with_pyzbar(image)
            all_detections.extend(direct_detections)
        
        # Procesar todas las detecciones
        for detection in all_detections:
            data = detection['data']
            det_type = detection['type']
            method = detection.get('method', 'unknown')
            bbox = detection.get('bbox')
            
            print(f"Detectado en imagen: {data} método: {method}")
            
            # Dibujar anotaciones
            annotated_image = self._draw_detection(annotated_image, detection)
            
            # Procesar código nuevo
            if data not in self.detected_barcodes:
                self.detected_barcodes.add(data)
                print(f"Nuevo código en imagen: {data}")
                
                # Buscar información del libro
                book_info = buscar_libro_openlibrary(data)
                
                info_list.append({
                    "isbn": data,
                    "barcode_type": det_type,
                    "detection_method": method,
                    "book_info": book_info,
                    "confidence": "high"  # Las imágenes estáticas tienen alta confianza
                })
                
                if book_info:
                    print(f"Libro encontrado: {book_info.get('titulo', 'Sin título')}")
                else:
                    print("No se encontró información del libro")
        
        return annotated_image, info_list
    
    def _draw_detection(self, image, detection):
        """Dibujar detección en la imagen"""
        try:
            data = detection['data']
            det_type = detection['type']
            method = detection.get('method', 'unknown')
            bbox = detection.get('bbox')
            
            if bbox:
                x1, y1, x2, y2 = bbox
            else:
                h, w = image.shape[:2]
                x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4
            
            # Color según el método
            if 'test' in method:
                color = (255, 0, 255)  # Magenta para pruebas
            elif 'pyzbar' in method:
                color = (0, 255, 0)    # Verde para pyzbar
            elif 'yolo' in method:
                color = (255, 165, 0)  # Naranja para YOLO
            else:
                color = (255, 255, 0)  # Cian para otros
            
            # Dibujar rectángulo más grueso para imágenes
            thickness = 3 if self.processing_mode == 'image' else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Texto principal con mejor formato
            text = f"{det_type}: {data}"
            font_scale = 0.8 if self.processing_mode == 'image' else 0.7
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # Texto principal
            cv2.putText(image, text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            
            # Texto del método (más pequeño)
            method_text = f"Método: {method}"
            cv2.putText(image, method_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        except Exception as e:
            print(f"Error dibujando detección: {e}")
        
        return image
    
    def process_frame_for_webrtc(self, frame):
        """Procesar frame principal (video/webcam)"""
        # Detectar si es una llamada desde procesamiento de imagen
        if hasattr(self, '_is_static_processing'):
            return self.process_image_static(frame)
        
        self.set_processing_mode('video')
        self.frame_count += 1
        info_list = []
        all_detections = []
        
        # Procesar menos frecuentemente en video para performance
        should_process = (self.frame_count % 5 == 0) or (self.frame_count < 10)
        
        if should_process:
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
        
        # Método 3: Crear detección de prueba cada 120 frames (para testing)
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
            
            print(f"Detectado: {data} método: {method}")
            
            # Dibujar en el frame
            frame = self._draw_detection(frame, detection)
            
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
    
    # Método específico para imágenes (wrapper)
    def process_single_image(self, image):
        """Método específico para procesar una sola imagen"""
        self._is_static_processing = True
        result = self.process_image_static(image)
        delattr(self, '_is_static_processing')
        return result
    
    def clear_detected_barcodes(self):
        """Limpiar códigos detectados"""
        self.detected_barcodes.clear()
        self.frame_count = 0
        print("Códigos detectados limpiados")
    
    def get_detection_stats(self):
        """Obtener estadísticas de detección"""
        return {
            "total_detected": len(self.detected_barcodes),
            "frames_processed": self.frame_count,
            "processing_mode": self.processing_mode,
            "yolo_available": YOLO_AVAILABLE
        }


