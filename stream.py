import streamlit as st
import numpy as np
from visio import VideoProcessor
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import time
import os
import joblib
import pandas as pd
import cv2
import hashlib
from PIL import Image

# Función para generar claves únicas
def generate_unique_key(base_key, book_data, extra_id=""):
    """Genera una clave única para widgets de Streamlit"""
    isbn = book_data.get('isbn', 'no_isbn')
    timestamp = str(int(time.time() * 1000))  # milisegundos
    data_hash = hashlib.md5(str(book_data).encode()).hexdigest()[:8]
    return f"{base_key}_{isbn}_{data_hash}_{extra_id}_{timestamp}"

# --- Cargar el modelo de recomendación ---
model_filename = 'book_recommender_model.joblib'
data_filename = 'book_data.joblib'

try:
    if os.path.exists(model_filename) and os.path.exists(data_filename):
        # El modelo optimizado es un diccionario con los libros más similares para cada libro.
        sparse_sim_dict = joblib.load(model_filename)
        df = joblib.load(data_filename)
        # Crear un mapeo de títulos a índices.
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        st.session_state.recommendation_model = {
            'sparse_sim_dict': sparse_sim_dict,
            'df': df,
            'indices': indices
        }
        st.sidebar.success("✅ Modelo de recomendación cargado correctamente.")
    else:
        st.session_state.recommendation_model = None
        st.sidebar.warning("⚠️ No se encontraron los archivos del modelo. Por favor, asegúrate de que estén en el directorio correcto.")
except Exception as e:
    st.session_state.recommendation_model = None
    st.sidebar.error(f"Error al cargar el modelo de recomendación: {str(e)}")


def get_content_based_recommendations(title):
    """
    Devuelve las 3 mejores recomendaciones de libros a partir de un título,
    usando el modelo de similitud optimizado.
    """
    if not st.session_state.recommendation_model:
        return "Modelo de recomendación no cargado."

    model_data = st.session_state.recommendation_model
    indices = model_data['indices']
    sparse_sim_dict = model_data['sparse_sim_dict']
    df = model_data['df']

    if title not in indices:
        return f"El libro '{title}' no se encontró en la base de datos para generar recomendaciones."

    # Obtener el índice del libro que coincide con el título.
    idx = indices[title]

    # Obtener las puntuaciones de similitud directamente del diccionario optimizado.
    sim_scores = sparse_sim_dict.get(idx, [])

    # Obtener los títulos de los 3 libros recomendados.
    recommended_titles = []
    # La lista ya está ordenada por similitud, así que tomamos los primeros 3.
    for score in sim_scores[:3]:
        book_index = score[0]
        recommended_titles.append(df.iloc[book_index]['title'])

    if not recommended_titles:
        return "No se encontraron libros similares para recomendar."
    
    response = "Recomendaciones basadas en el contenido:\n"
    for i, rec_title in enumerate(recommended_titles, 1):
        response += f"{i}. {rec_title}\n"
    
    return response

# Título de la aplicación Streamlit
st.title("Escáner de Códigos de Barras de Libros")

# Sidebar para debug
st.sidebar.header("Estado del Sistema")
st.sidebar.write("Información de debug aparecerá aquí...")

# Initialize the processor in session state
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

# Initialize detected books list
if 'detected_books' not in st.session_state:
    st.session_state.detected_books = []

# --- Lógica para la cámara web ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.processor = VideoProcessor()
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Procesar el cuadro y obtener las anotaciones y la lista de información
        processed_frame, info_list = self.processor.process_frame_for_webrtc(img)
        
        # Agregar nuevos libros detectados sin reiniciar
        if info_list:
            with self.lock:
                # Filtrar cualquier elemento None antes de procesar
                for book_info in [b for b in info_list if b is not None and b.get('isbn')]:
                    isbn = book_info['isbn']
                    # Solo agregar si no existe ya
                    if not any(book['isbn'] == isbn for book in st.session_state.detected_books):
                        st.session_state.detected_books.append(book_info)
        
        return processed_frame

# --- Interfaz de usuario para la selección de la fuente de video ---
st.header("Selecciona una Fuente de Entrada")
source_option = st.selectbox(
    "Elige una opción para escanear:",
    ("Subir una Imagen", "Cámara Web en Vivo", "Subir un Archivo de Video")
)

# --- Conditional Logic ---
if source_option == "Subir una Imagen":
    st.info("Sube una imagen que contenga códigos de barras de libros para detectar y obtener recomendaciones.")
    
    uploaded_image = st.file_uploader(
        "Elige un archivo de imagen (.jpg, .jpeg, .png, .bmp)", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_image is not None:
        # Mostrar la imagen subida
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Imagen Original")
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            st.subheader("Procesamiento")
            with st.spinner("Procesando imagen..."):
                # Convertir imagen a formato OpenCV
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    cv_image = image_array
                
                # Procesar la imagen
                processed_frame, info_list = st.session_state.processor.process_frame_for_webrtc(cv_image)
                
                # Mostrar imagen procesada
                st.image(processed_frame, caption="Imagen procesada", channels="BGR", use_column_width=True)
        
        # Procesar resultados
        books_found_in_image = []
        if info_list:
            for book_info in info_list:
                if book_info is not None and isinstance(book_info, dict) and book_info.get('isbn'):
                    isbn = book_info['isbn']
                    # Solo agregar libros únicos
                    if not any(book and isinstance(book, dict) and book.get('isbn') == isbn for book in books_found_in_image):
                        books_found_in_image.append(book_info)
                        
                        # También agregar a la lista global
                        if not any(b and b.get('isbn') == isbn for b in st.session_state.detected_books):
                            st.session_state.detected_books.append(book_info)
        
        # Mostrar resultados
        if books_found_in_image:
            st.success(f"🎉 Se detectaron {len(books_found_in_image)} código(s) de barras!")
            
            for i, book in enumerate(books_found_in_image):
                # Verificar que book no sea None y sea un diccionario válido
                if book is None or not isinstance(book, dict):
                    continue
                
                # Título para el expander
                try:
                    title_for_expander = book.get('book_info', {}).get('titulo', book.get('isbn', 'Libro sin título'))
                    if not title_for_expander:
                        title_for_expander = f"Libro {i+1}"
                except (AttributeError, TypeError):
                    title_for_expander = f"Libro {i+1}"
                
                with st.expander(f"📖 Libro {i+1}: {title_for_expander}", expanded=True):
                    # Dividir en columnas: información (izquierda) y recomendaciones (derecha)
                    col_info, col_recs = st.columns([1, 1])
                    
                    with col_info:
                        st.markdown("### 📚 Información del Libro")
                        try:
                            if book.get('book_info') and isinstance(book.get('book_info'), dict):
                                book_info_dict = book['book_info']
                                st.write(f"**Título:** {book_info_dict.get('titulo', 'N/A')}")
                                st.write(f"**Autores:** {book_info_dict.get('autores', 'N/A')}")
                                st.write(f"**Editorial:** {book_info_dict.get('editorial', 'N/A')}")
                                st.write(f"**ISBN:** {book.get('isbn', 'N/A')}")
                            else:
                                st.write(f"**ISBN:** {book.get('isbn', 'N/A')}")
                                st.info("ℹ️ Información no encontrada en OpenLibrary")
                        except (AttributeError, TypeError, KeyError) as e:
                            st.error(f"Error al procesar información del libro: {str(e)}")
                            st.write(f"**ISBN:** {book.get('isbn', 'N/A') if book else 'Error'}")
                    
                    with col_recs:
                        st.markdown("### 🤖 Recomendaciones")
                        
                        # Obtener título del libro para recomendaciones
                        book_title = None
                        try:
                            book_info_check = book.get('book_info')
                            if book_info_check and isinstance(book_info_check, dict):
                                book_title = book_info_check.get('titulo')
                        except (AttributeError, TypeError):
                            book_title = None
                        
                        if not book or not book.get('isbn'):
                            st.error("❌ Datos del libro inválidos")
                            continue
                            
                        rec_key = f"image_rec_{book['isbn']}"
                        
                        if book_title and st.session_state.get('recommendation_model'):
                            # Generar recomendaciones automáticamente si no existen
                            if rec_key not in st.session_state:
                                with st.spinner("🔍 Generando recomendaciones..."):
                                    try:
                                        recommendations = get_content_based_recommendations(book_title)
                                        st.session_state[rec_key] = recommendations
                                    except Exception as e:
                                        st.error(f"❌ Error generando recomendaciones: {str(e)}")
                                        st.session_state[rec_key] = "Error al generar recomendaciones"
                            
                            # Mostrar recomendaciones
                            st.markdown(st.session_state[rec_key])
                            
                            # Botón para regenerar con clave única
                            try:
                                if st.button("🔄 Regenerar Recomendaciones", key=generate_unique_key("regenerate_image", book, str(i))):
                                    with st.spinner("🔄 Regenerando..."):
                                        try:
                                            new_recs = get_content_based_recommendations(book_title)
                                            st.session_state[rec_key] = new_recs
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error regenerando: {str(e)}")
                            except Exception as e:
                                st.error(f"❌ Error creando botón: {str(e)}")
                        else:
                            st.info("ℹ️ Título del libro no disponible para generar recomendaciones.")
                            
                            # Opción manual si no hay título
                            if not book_title:
                                st.markdown("**Búsqueda manual:**")
                                manual_title = st.text_input(
                                    "Ingresa el título manualmente:", 
                                    key=f"manual_title_{book['isbn']}",
                                    placeholder="Ej: El nombre del viento"
                                )
                                
                                if manual_title and st.button("🔍 Buscar", key=f"manual_search_{book['isbn']}"):
                                    with st.spinner("Buscando recomendaciones..."):
                                        try:
                                            manual_recs = get_content_based_recommendations(manual_title)
                                            st.session_state[rec_key] = manual_recs
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ No se encontraron recomendaciones para '{manual_title}'")
        else:
            st.warning("⚠️ No se detectaron códigos de barras en la imagen.")
            st.info("""
            **💡 Consejos para mejor detección:**
            - Asegúrate de que el código de barras esté bien enfocado
            - Usa buena iluminación sin reflejos
            - El código debe ser claramente visible
            - Prueba con diferentes ángulos si no funciona
            """)
    
    else:
        st.info("👆 Por favor, sube una imagen para comenzar el análisis.")

elif source_option == "Cámara Web en Vivo":
    st.info("Iniciando la cámara web. Permite el acceso en el navegador para comenzar a escanear.")
    
    # Inicia la transmisión de la cámara web
    webrtc_streamer(
        key="webcam_streamer",
        video_processor_factory=VideoTransformer,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )
    
    # Botón para limpiar códigos detectados
    if st.button("Limpiar códigos detectados"):
        st.session_state.processor.clear_detected_barcodes()
        st.session_state.detected_books = []
        st.success("Códigos detectados limpiados")

    # Mostrar información de libros detectados
    if st.session_state.detected_books:
        st.success(f"Códigos de barras detectados: {len(st.session_state.detected_books)}")
        
        for idx, info in enumerate(st.session_state.detected_books):
            with st.expander(f"Libro {idx+1}: {info.get('book_info', {}).get('titulo', info['isbn']) if info.get('book_info') else info['isbn']}", expanded=False):
                
                # Dividir en dos columnas: información del libro (izquierda) y recomendaciones (derecha)
                col_info, col_recs = st.columns([1, 1])
                
                with col_info:
                    st.markdown("### 📖 Información del Libro")
                    if info.get('book_info'):
                        st.write(f"**Título:** {info['book_info']['titulo']}")
                        st.write(f"**Autores:** {info['book_info']['autores']}")
                        st.write(f"**Editorial:** {info['book_info']['editorial']}")
                    else:
                        st.warning("Información del libro no encontrada en OpenLibrary")
                    
                    st.write(f"**ISBN:** {info['isbn']}")
                    st.write(f"**Método de detección:** {info.get('detection_method', 'N/A')}")
                
                with col_recs:
                    st.markdown("### 🤖 Recomendaciones")
                    
                    # Se ha mejorado la forma de obtener el título para evitar el error 'NoneType'
                    book_info = info.get('book_info')
                    book_title = book_info.get('titulo') if book_info else None
                    
                    rec_key = f"recommendations_{info['isbn']}"
                    
                    if book_title and st.session_state.get('recommendation_model'):
                        # Verificar si ya tenemos recomendaciones
                        if rec_key not in st.session_state:
                            # Generar recomendaciones automáticamente
                            with st.spinner("Generando recomendaciones..."):
                                recommendations = get_content_based_recommendations(book_title)
                                st.session_state[rec_key] = recommendations
                        
                        # Mostrar recomendaciones
                        st.markdown(st.session_state[rec_key])
                        
                        # Botón para regenerar con clave única
                        if st.button("🔄 Regenerar", key=generate_unique_key("regenerate_webcam", info, str(idx)), size="small"):
                            with st.spinner("Regenerando..."):
                                new_recs = get_content_based_recommendations(book_title)
                                st.session_state[rec_key] = new_recs
                                st.rerun()
                    else:
                        st.info("Título del libro no disponible para generar recomendaciones.")

elif source_option == "Subir un Archivo de Video":
    uploaded_file = st.file_uploader("Elige un archivo de video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        temp_file_path = "temp_video.mp4"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Crear placeholders que se actualizarán
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
        info_container = st.empty()
        
        # Variable para controlar el procesamiento
        stop_processing = False
        
        # Botón para detener procesamiento
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⏹️ Detener"):
                stop_processing = True
        
        st.write("Procesando video en tiempo real...")

        # Process video using OpenCV directly
        cap = cv2.VideoCapture(temp_file_path)
        
        if not cap.isOpened():
            st.error("No se pudo abrir el archivo de video.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            books_found_in_video = []
            
            while True and not stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Actualizar progreso
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_placeholder.progress(progress, text=f"Frame {frame_count}/{total_frames}")
                
                # Process every 10th frame
                if frame_count % 10 == 0:
                    # Process the frame
                    processed_frame, info_list = st.session_state.processor.process_frame_for_webrtc(frame)
                    
                    # Display the frame
                    video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    # Process new detections
                    if info_list:
                        for book_info in info_list:
                            # Se agrega una verificación para evitar agregar objetos None
                            if book_info is not None and isinstance(book_info, dict) and book_info.get('isbn'):
                                isbn = book_info['isbn']
                                # Solo agregar libros únicos
                                if not any(book and isinstance(book, dict) and book.get('isbn') == isbn for book in books_found_in_video):
                                    books_found_in_video.append(book_info)
                            else:
                                # Log error pero continúa procesando
                                print(f"Elemento inválido en info_list: {book_info}")
                                continue
                    
                    # Mostrar información acumulada sin reiniciar
                    with info_container.container():
                        if books_found_in_video:
                            st.subheader(f"📚 Libros encontrados: {len(books_found_in_video)}")
                            
                            for i, book in enumerate(books_found_in_video):
                                # Verificar que book no sea None y sea un diccionario válido
                                if book is None or not isinstance(book, dict):
                                    continue
                                
                                # Se ha mejorado la forma de obtener el título para el expander
                                try:
                                    title_for_expander = book.get('book_info', {}).get('titulo', book.get('isbn', 'Libro sin título'))
                                    if not title_for_expander:
                                        title_for_expander = f"Libro {i+1}"
                                except (AttributeError, TypeError):
                                    title_for_expander = f"Libro {i+1}"
                                
                                with st.expander(f"Libro {i+1}: {title_for_expander}", expanded=True):
                                    
                                    # Dividir en columnas: información (izquierda) y recomendaciones (derecha)
                                    col_info, col_recs = st.columns([1, 1])
                                    
                                    with col_info:
                                        st.markdown("### 📖 Información")
                                        # Verificar que book_info existe y es válido
                                        try:
                                            if book.get('book_info') and isinstance(book.get('book_info'), dict):
                                                book_info_dict = book['book_info']
                                                st.write(f"**Título:** {book_info_dict.get('titulo', 'N/A')}")
                                                st.write(f"**Autores:** {book_info_dict.get('autores', 'N/A')}")
                                                st.write(f"**Editorial:** {book_info_dict.get('editorial', 'N/A')}")
                                                st.write(f"**ISBN:** {book.get('isbn', 'N/A')}")
                                            else:
                                                st.write(f"**ISBN:** {book.get('isbn', 'N/A')}")
                                                st.warning("Información no encontrada en OpenLibrary")
                                        except (AttributeError, TypeError, KeyError) as e:
                                            st.error(f"Error al procesar información del libro: {str(e)}")
                                            st.write(f"**ISBN:** {book.get('isbn', 'N/A') if book else 'Error'}")
                                    
                                    with col_recs:
                                        st.markdown("### 🤖 Recomendaciones")
                                        
                                        # Se ha mejorado la forma de obtener el título para evitar el error 'NoneType'
                                        book_title = None
                                        try:
                                            book_info = book.get('book_info')
                                            if book_info and isinstance(book_info, dict):
                                                book_title = book_info.get('titulo')
                                        except (AttributeError, TypeError):
                                            book_title = None
                                        
                                        if not book or not book.get('isbn'):
                                            st.error("Datos del libro inválidos")
                                            continue
                                            
                                        rec_key = f"video_rec_{book['isbn']}"
                                        
                                        if book_title and st.session_state.get('recommendation_model'):
                                            # Generar recomendaciones automáticamente si no existen
                                            if rec_key not in st.session_state:
                                                with st.spinner("Generando..."):
                                                    try:
                                                        recommendations = get_content_based_recommendations(book_title)
                                                        st.session_state[rec_key] = recommendations
                                                    except Exception as e:
                                                        st.error(f"Error generando recomendaciones: {str(e)}")
                                                        st.session_state[rec_key] = "Error al generar recomendaciones"
                                            
                                            # Mostrar recomendaciones
                                            st.markdown(st.session_state[rec_key])
                                            
                                            # Botón para regenerar con clave única
                                            try:
                                                if st.button("🔄 Regenerar", key=generate_unique_key("regenerate_video", book, str(i))):
                                                    with st.spinner("Regenerando..."):
                                                        try:
                                                            new_recs = get_content_based_recommendations(book_title)
                                                            st.session_state[rec_key] = new_recs
                                                            st.rerun()
                                                        except Exception as e:
                                                            st.error(f"Error regenerando: {str(e)}")
                                            except Exception as e:
                                                st.error(f"Error creando botón: {str(e)}")
                                        else:
                                            st.info("Título del libro no disponible para generar recomendaciones.")
                
                # Pausa pequeña para no sobrecargar
                time.sleep(0.05)
            
            cap.release()
            
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            progress_placeholder.empty()
            
            if books_found_in_video:
                st.success(f"✅ Procesamiento completado. {len(books_found_in_video)} libro(s) único(s) encontrado(s).")
                
                # Agregar libros del video a la lista global
                for book in books_found_in_video:
                    if not any(b.get('isbn') == book.get('isbn') for b in st.session_state.detected_books):
                        st.session_state.detected_books.append(book)
            else:
                st.info("No se detectaron códigos de barras en el video.")
    else:
        st.info("Por favor, sube un archivo de video para comenzar el escaneo.")

# Sección de recomendaciones consolidada
if st.session_state.detected_books:
    st.markdown("---")
    st.header("📚 Panel de Recomendaciones")
    
    books_with_recommendations = []
    for book in st.session_state.detected_books:
        isbn = book.get('isbn')
        if isbn and (f"recommendations_{isbn}" in st.session_state or f"video_rec_{isbn}" in st.session_state):
            books_with_recommendations.append(book)
    
    if books_with_recommendations:
        st.markdown(f"*Recomendaciones generadas para {len(books_with_recommendations)} libro(s)*")
        
        for book in books_with_recommendations:
            isbn = book.get('isbn')
            title = book.get('book_info', {}).get('titulo', isbn) if book.get('book_info') else isbn
            
            # Obtener recomendaciones desde session_state
            recommendations_text = None
            if f"recommendations_{isbn}" in st.session_state:
                recommendations_text = st.session_state[f"recommendations_{isbn}"]
            elif f"video_rec_{isbn}" in st.session_state:
                recommendations_text = st.session_state[f"video_rec_{isbn}"]
            
            if recommendations_text:
                with st.expander(f"📚 Recomendaciones para: {title}", expanded=False):
                    # Información del libro base
                    if book.get('book_info'):
                        st.write(f"**Libro base:** {book['book_info']['titulo']}")
                        st.write(f"**Autor:** {book['book_info']['autores']}")
                        st.write(f"**ISBN:** {isbn}")
                    else:
                        st.write(f"**ISBN:** {isbn}")
                    
                    st.divider()
                    
                    # Mostrar recomendaciones
                    st.markdown(recommendations_text)
                    
                    # Botones de acción
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Regenerar", key=generate_unique_key("regenerate_consolidated", book)):
                            book_info_check = book.get('book_info')
                            book_title = book_info_check.get('titulo') if book_info_check else None
                            with st.spinner("Regenerando recomendaciones..."):
                                if book_title:
                                    new_recs = get_content_based_recommendations(book_title)
                                    # Actualizar en ambas posibles ubicaciones
                                    if f"recommendations_{isbn}" in st.session_state:
                                        st.session_state[f"recommendations_{isbn}"] = new_recs
                                    if f"video_rec_{isbn}" in st.session_state:
                                        st.session_state[f"video_rec_{isbn}"] = new_recs
                                    st.rerun()
                                else:
                                    st.warning("Título del libro no disponible para regenerar recomendaciones.")
                    
                    with col2:
                        # Botón de exportar con clave única
                        export_text = f"RECOMENDACIONES DE LIBROS\n"
                        export_text += f"{'='*40}\n\n"
                        export_text += f"Libro: {title}\n"
                        export_text += f"ISBN: {isbn}\n"
                        if book.get('book_info'):
                            export_text += f"Autor: {book['book_info']['autores']}\n"
                        export_text += f"\nRecomendaciones:\n{'-'*20}\n\n"
                        export_text += recommendations_text
                        
                        st.download_button(
                            "📄 Exportar",
                            export_text,
                            file_name=f"recomendaciones_{isbn}.txt",
                            mime="text/plain",
                            key=generate_unique_key("export", book)
                        )
    else:
        st.info("Haz clic en 'Obtener recomendaciones' en cualquier libro detectado para ver las sugerencias aquí.")

# Sidebar con estadísticas actualizadas
st.sidebar.markdown("### 📊 Estadísticas")
if st.session_state.detected_books:
    total_books = len(st.session_state.detected_books)
    books_with_recs = len([b for b in st.session_state.detected_books
                           if b and b.get('isbn') and (f"recommendations_{b['isbn']}" in st.session_state or f"video_rec_{b['isbn']}" in st.session_state)])
    
    st.sidebar.write(f"📚 Total libros detectados: {total_books}")
    st.sidebar.write(f"🤖 Con recomendaciones: {books_with_recs}")
    
    if books_with_recs > 0:
        st.sidebar.write(f"📋 Panel de recomendaciones activo")
    
    # Mostrar últimos libros detectados
    st.sidebar.markdown("### 📖 Últimos detectados:")
    for book in st.session_state.detected_books[-3:]:  # Últimos 3
        if book and book.get('book_info') and book['book_info'].get('titulo'):
            title = book['book_info']['titulo']
        elif book and book.get('isbn'):
            title = book['isbn']
        else:
            title = "N/A"
        st.sidebar.write(f"• {title[:30]}...")
else:
    st.sidebar.write("Aún no hay libros detectados")

st.sidebar.markdown("### 🔧 Herramientas")
if st.sidebar.button("🗑️ Limpiar todo"):
    st.session_state.detected_books = []
    if 'processor' in st.session_state:
        st.session_state.processor.clear_detected_barcodes()
    # Limpiar también las recomendaciones
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith('recommendations_') or k.startswith('video_rec_')]
    for key in keys_to_remove:
        del st.session_state[key]
    st.success("Todos los datos limpiados")
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**💡 Consejos para mejor detección:**
- Mantén el código de barras bien iluminado
- Asegúrate de que el código esté enfocado
- Mantén el código paralelo a la cámara
- Evita reflejos y sombras sobre el código
""")
