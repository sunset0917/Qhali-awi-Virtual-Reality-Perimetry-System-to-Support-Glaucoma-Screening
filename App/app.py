import firebase_admin
from firebase_admin import credentials, firestore, storage
from flask import Flask, jsonify, Response
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from io import BytesIO
import os
import pandas as pd
import numpy as np
from flask_cors import CORS
import cv2
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("")
        firebase_admin.initialize_app(cred, {
            'storageBucket': ''
        })
        db = firestore.client() 
        print("Firebase inicializado correctamente.")
    except Exception as e:
        print(f"Error al inicializar Firebase: {str(e)}")
        exit(1)


app = Flask(__name__)
POST_FOLDER = 'Postprocesamiento'
os.makedirs(POST_FOLDER, exist_ok=True)
output_file = os.path.join(POST_FOLDER, "fixation23.txt")
RESULT_FILE="posiciones_con_perdida.txt"
is_tracking = False


stream_url = "http://192.168.18.27:81/stream"

pupil_positions = []  
fixation_loss_total = 0  

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("No se puede abrir el video")
    exit()


def extract_pupil(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=33, 
        minRadius=5, 
        maxRadius=50
    )
    pupil_center = None
    thres = np.zeros_like(gray)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            if mean_intensity < 150:  # Umbral ajustable
                pupil_center = (x, y)
                thres = cv2.circle(thres, (x, y), r, 255, -1)
                break
    return pupil_center, thres

def draw_cross(frame, center):

    if center:
        x, y = center
        cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2)

def update_positions(positions, new_position, max_length=10):

    positions.append(new_position)
    if len(positions) > max_length:
        positions.pop(0)
    return positions

def calculate_fixation_loss(positions, threshold):

    fixation_loss_count = 0
    for i in range(1, len(positions)):
        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance > threshold:
            fixation_loss_count += 1

        if fixation_loss_count > 0:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(output_file, "a") as f:
                f.write(f"{current_time}\n")
    return fixation_loss_count

def upload_results_to_firebase():

    bucket = storage.bucket()
    blob = bucket.blob(f'Postprocesamiento/fixation.txt')
    blob.upload_from_filename(output_file)
    print("Archivo fixation.txt subido a Firebase en la carpeta Postprocesamiento.")


def leer_estimulos(filename, lineas_a_omitir):
    estimulos = []
    with open(filename, 'r') as file:
        for i, linea in enumerate(file):
            if i not in lineas_a_omitir and linea.strip():
                partes = linea.strip().split(': ', 1)
                if len(partes) == 2 and partes[0].startswith("Posición"):
                    try:
                        posicion = int(partes[0].split()[1])
                        tiempo = partes[1].strip().split()[1]
                        tiempo_obj = datetime.strptime(tiempo, "%H:%M:%S.%f").time()
                        estimulos.append((posicion, tiempo_obj))
                    except (ValueError, IndexError):
                        pass
    return estimulos


def leer_fijaciones(filename, lineas_a_omitir):
    fijaciones = []
    with open(filename, 'r') as file:
        for i, linea in enumerate(file):
            if i not in lineas_a_omitir and linea.strip():
                try:
                    tiempo = linea.strip().split()[1]
                    tiempo_obj = datetime.strptime(tiempo, "%H:%M:%S.%f").time()
                    fijaciones.append(tiempo_obj)
                except (ValueError, IndexError):
                    pass
    return fijaciones


def comparar_tiempos(fijaciones, estimulos, archivo_salida):
    matches = []
    for fijacion in fijaciones:
        for posicion, tiempo in estimulos:
            diferencia = abs(
                (datetime.combine(datetime.min, tiempo) - datetime.combine(datetime.min, fijacion)).total_seconds()
            )
            if diferencia < 1:
                matches.append(posicion)
                break
    with open(archivo_salida, "w") as f:
        f.write("Posiciones con pérdida de fijación:\n")
        for match in matches:
            f.write(f"{match}\n")
def generate_adjusted_points():
                points = []
                rows = [4, 6, 8, 9, 9, 8, 6, 4]
                y_positions = np.arange(3, -5, -1)  
                for i, num_points in enumerate(rows):
                
                    if num_points == 9:
                        x_positions = np.linspace(-((num_points - 1) / 2) - 0.5, (num_points - 1) / 2 - 0.5, num_points)
                    else:
                        x_positions = np.linspace(-((num_points - 1) / 2), (num_points - 1) / 2, num_points)
                    y = y_positions[i]
                    points.extend([(x, y) for x in x_positions])
                return np.array(points)

@app.route('/')
def index():
    return 

@app.route('/procesar', methods=['POST'])
def procesar_archivos():
    bucket = storage.bucket()
    folder = 'Postprocesamiento/' 


    estimulos_path = os.path.join(POST_FOLDER, 'timestamps.txt')
    fijaciones_path = os.path.join(POST_FOLDER, 'fixation.txt')

    archivos_descargados = {
        "timestamps": False,
        "fixation": False
    }

    for blob in bucket.list_blobs(prefix=folder):
        print(f"Archivo encontrado: {blob.name}")
        if blob.name.endswith('timestamps.txt'):
            blob.download_to_filename(estimulos_path)
            archivos_descargados["timestamps"] = True
            print(f"Archivo descargado: {estimulos_path}")
        elif blob.name.endswith('fixation.txt'):
            blob.download_to_filename(fijaciones_path)
            archivos_descargados["fixation"] = True
            print(f"Archivo descargado: {fijaciones_path}")

    if not archivos_descargados["timestamps"] or not archivos_descargados["fixation"]:
        return "<h1>Error: No se pudieron descargar ambos archivos necesarios desde Firebase.</h1>"

    estimulos = leer_estimulos(estimulos_path, range(7))  
    fijaciones = leer_fijaciones(fijaciones_path, [0])  
    comparar_tiempos(fijaciones, estimulos, RESULT_FILE)

    result_blob = bucket.blob(f"{folder}resultados.txt")
    result_blob.upload_from_filename(RESULT_FILE)
    print(f"Archivo de resultados subido a Firebase en la carpeta {folder}")
    return 

@app.route('/descargar')
def descargar_resultados():
    return send_file(RESULT_FILE, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global cap, pupil_positions, fixation_loss_total
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            pupil_center, thres = extract_pupil(frame)
            if pupil_center is not None:
                draw_cross(frame, pupil_center)
                pupil_positions = update_positions(pupil_positions, pupil_center)
                fixation_loss_count = calculate_fixation_loss(pupil_positions, threshold=50)
                fixation_loss_total += fixation_loss_count
                if fixation_loss_count > 0:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    with open(output_file, "a") as f:
                        f.write(f"{current_time}\n")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fixation_loss_counter', methods=['GET'])
def get_fixation_loss_counter():
    return jsonify({"fixation_loss_counter": fixation_loss_total})

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global is_tracking, fixation_loss_total
    is_tracking = True
    fixation_loss_total = 0  
    return jsonify({"message": "Tracking started"})

@app.route('/pause_tracking', methods=['POST'])
def pause_tracking():
    global is_tracking
    is_tracking = False
    return jsonify({"message": "Tracking paused"})


@app.route('/end_tracking', methods=['POST'])
def end_tracking():

    global is_tracking
    is_tracking = False 
    try:
        upload_results_to_firebase()  
        return jsonify({"message": "Tracking finalizado y resultados subidos a Firebase."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results', methods=['GET'])
def get_input_data():
    try:
 
        doc_ref = db.collection('lastpatient').document('lastpatient')
        doc = doc_ref.get()

        if doc.exists:
            patient_data = doc.to_dict()
            ID = patient_data.get("id")
            FDN = patient_data.get("dob")
            sexo = patient_data.get("gender")
            edad = patient_data.get("age")
            estimulo = patient_data.get("examConfig", {}).get("stimulusSize")
            patron = patient_data.get("examConfig", {}).get("pattern")
            ojo_real = patient_data.get("examConfig", {}).get("eye")
            ojo = "OD"
            diametro = 4 
            apellido = patient_data.get("lastName")
        else:
            return jsonify({"error": "No se encontró el documento del paciente."}), 404
               
        bucket = storage.bucket()
        if not bucket:
            raise Exception("No se pudo acceder al bucket de almacenamiento.")
    
        file_name = f'Postprocesamiento/{ID}_{apellido}.txt'
        blob = bucket.blob(file_name)
        input_data = blob.download_as_text()
        
        
        def extract_numbers_from_positions_and_negatives(data):
            lines = data.splitlines()
            numbers = []
            falsos_negativos = None

            for line in lines:
                if line.startswith("Posición"):
                    try:
                        _, value = line.split(":")
                        numbers.append(int(value.strip()))
                    except ValueError:
                        continue
                
                if line.startswith("Falsos negativos"):
                    try:
                        _, value = line.split(":")
                        falsos_negativos = int(value.strip())
                    except ValueError:
                        continue
            
            return numbers,falsos_negativos

        data_array, falsos_negativos = extract_numbers_from_positions_and_negatives(input_data)
        print(data_array)
        
        if len(data_array) == 8:

            adjusted_points = generate_adjusted_points()
            adjusted_points[:, 1] += 0.5

            selected_points = np.array([
                (-0.5, 3.5),       # Segundo punto de la primera fila
                (-1.5, 1.5),    # Tercer punto de la tercera fila
                (1.5, 1.5),     # Sexto punto de la tercera fila
                (3.5, 0.5),     # Último punto de la cuarta fila
                (-3.5, -0.5),   # Segundo punto de la quinta fila
                (-1.5, -1.5),      # Tercer punto de la quinta fila
                (1.5, -1.5),    # Sexto punto de la sexta fila
                (0.5, -3.5),      # Tercer punto de la última fila
            ])
        
            
            # Crear una malla de puntos para interpolar
            grid_x, grid_y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(4, -4, 100))

            # Interpolación de la malla con los puntos seleccionados y los valores aleatorios
            grid_z = griddata(selected_points, data_array, (grid_x, grid_y), method='nearest')

            # Para obtener los valores interpolados en los puntos de la matriz completa (no solo en los seleccionados)
            interpolated_values = griddata(selected_points, data_array, adjusted_points, method='nearest')
            interpolated_values = interpolated_values.astype(int)
            
            # Imprimir el array de los valores interpolados para los puntos de la matriz completa
            #print("Valores interpolados para los puntos de la matriz:")
            #print(interpolated_values)
            if ojo_real=="OI":
                # Reemplazar los valores de las posiciones 25, 26, 34 y 35 con valores aleatorios entre 0 y 5
                np.random.seed(42)  # Fijar semilla para reproducibilidad
                indices_to_replace = [20, 21, 29, 30]
                interpolated_values[indices_to_replace] = np.random.randint(0, 6, size=len(indices_to_replace))
            else:
                np.random.seed(42)  # Fijar semilla para reproducibilidad
                indices_to_replace = [24, 25, 33, 34]
                interpolated_values[indices_to_replace] = np.random.randint(0, 6, size=len(indices_to_replace))
            data_array = interpolated_values
        print(data_array)
        from PyVisualFields import vfprogression, visualFields
        
        # Genera la imagen y guarda en un archivo temporal
        vfprogression.plotValues(data_array, title='Test results', save=True, filename="test_results", fmt='png')

        #Ajustar tabla
        # Define los encabezados para las nuevas columnas y sus valores iniciales
        extra_columns = {
            "id": [ID],
            "eye": [ojo],
            "date": [pd.to_datetime("2008-08-13")], 
            "time": ["00:00:00"],
            "age": [edad],
            "type": ["pwg"],
            "fpr": [0],
            "fnr": [0.0],
            "fl": [0.00],
            "duration": ["00:00:00"]
        }
        # Crea una lista de encabezados de columna desde l1 a l54
        column_headers = [f"l{i}" for i in range(1, 55)]

        # Convierte el array de datos en un DataFrame de una sola fila con columnas l1-l54
        df_values = pd.DataFrame([data_array], columns=column_headers)

        # Convierte las columnas adicionales en un DataFrame y concatena a la izquierda de df_values
        df_extra = pd.DataFrame(extra_columns)
        df = pd.concat([df_extra, df_values], axis=1)
        
        df_td, df_tdp, df_gi, df_gip, df_pd, df_pdp, gh = visualFields.getallvalues(df)
        
        #TOTAL DEVIATION
        l_columns = df_td.loc[:, 'l1':'l54']
        l_array = l_columns.values
        patient_td = np.round(l_array[0,:], 1)
              
        vfprogression.plotValues(patient_td, title= 'Total Deviation',save=True, filename='total_deviation', fmt='png')
        
        #TOTAL DEVIATION PROBABILITY
        l_columns_tdp = df_tdp.loc[:, 'l1':'l54']
        l_array_tdp = l_columns_tdp.values
        patient_tdp = np.round(l_array_tdp[0,:], 1)
        
        vfprogression.plotProbabilities(patient_tdp, title= 'Total Deviation Probablity', save=True, filename='total_deviation_probability', fmt='png')  

        
        #PATTERN DEVIATION
        l_columns_pd = df_pd.loc[:, 'l1':'l54']
        l_array_pd = l_columns_pd.values
        patient_pd = np.round(l_array_pd[0,:], 1)
        
        vfprogression.plotValues(patient_pd, title='Pattern deviation', save=True, filename='pd', fmt='png')
        
        #PATTERN DEVIATION PROBABILITY
        l_columns_pdp = df_pdp.loc[:, 'l1':'l54']
        l_array_pdp = l_columns_pdp.values
        patient_pdp = np.round(l_array_pdp[0,:], 1)
        
        vfprogression.plotProbabilities(patient_pdp, title='Pattern deviation probability', save=True, filename='pdp', fmt='png')

        #ESCALA DE GRISES
        visualFields.vfplot(df, type='s', save=True, filename='greyscale_results', fmt='png') # types: s, td,pd, tds, pds

        # Crear un buffer en memoria para el PDF
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)

        # Añadir título al PDF
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 800, "Resultados del test de campo visual")
        c.setFont("Helvetica", 10)
        c.drawString(50, 780, f"ID: {ID}")
        c.drawString(50, 760, f"FDN: {FDN}")
        c.drawString(50, 740, f"Edad: {edad}")
        c.drawString(50, 720, f"Sexo: {sexo}")
        c.drawString(50, 700, f"Monitor de fijación: Eye-tracking")
        c.drawString(50, 680, f"Objetivo de fijación: Central")

        # Resultados del examen como array
        perdida_fijacion = fixation_loss_counter
        falsos_positivos = 0
        falsos_negativos = round((falsos_negativos/54)*100)
        duracion = 4.5  # minutos
        
        c.drawString(250, 780, f"Pérdidas de fijación: {perdida_fijacion}/10")
        c.drawString(250, 760, f"Errores falsos pos.: {falsos_positivos}%")
        c.drawString(250, 740, f"Errores falsos neg.: {falsos_negativos}%")
        c.drawString(250, 720, f"Duración: {duracion} min")
        c.drawString(420, 780, f"Estímulo: {estimulo}, Blanco")
        c.drawString(420, 760, f"Fondo: 31.5 asb")
        c.drawString(420, 740, f"Estrategia: Supraumbral Doble Cruce")
        c.drawString(420, 720, f"Patrón: {patron}")
        c.drawString(420, 700, f"Diámetro de la pupila: {diametro} mm")
        c.drawString(420, 680, f"Ojo: {ojo_real}")

        # Añadir la imagen generada al PDF
        c.drawImage("test_results.png",50,450, width=200, height=200)
        c.drawImage("greyscale_results.png",300,450, width=200, height=200)
        c.drawImage("total_deviation.png",50,250, width=200, height=200)
        c.drawImage("pd.png",300,250, width=200, height=200)
        c.drawImage("total_deviation_probability.png",50,50, width=200, height=200)
        c.drawImage("pdp.png",300,50, width=200, height=200)

        # Finaliza el PDF
        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # Sube el PDF a Firebase Storage
        file_name = f'uploads/{ID}.pdf'
        pdf_blob = bucket.blob(file_name)
        pdf_blob.upload_from_file(pdf_buffer, content_type='application/pdf')

        # Asegura que el archivo PDF sea público y obtiene la URL
        pdf_blob.make_public()
        pdf_public_url = pdf_blob.public_url
        
        # Estructura de datos para Firestore
        document_data = {
            "name": f'{ID}.pdf',  
            "patientId": ID,  
            "url": pdf_public_url
        }

        # Guarda el documento en Firestore
        db.collection('pdfUploads').add(document_data)  
        
        # Limpia el archivo de imagen temporal
        if os.path.exists("test_results.png"):
            os.remove("test_results.png")

        return jsonify({
            "message": "PDF generado y subido exitosamente",
            "pdf_url": pdf_public_url
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error al generar o subir el PDF: {str(e)}"}), 500

@app.route('/processing', methods=['GET', 'POST'])
def procesar_archivos():
    bucket = storage.bucket()
    folder = 'Postprocesamiento/'  # Carpeta en Firebase Storage

    # Descargar archivos desde Firebase Storage
    estimulos_path = os.path.join(POST_FOLDER, 'timestamps.txt')
    fijaciones_path = os.path.join(POST_FOLDER, 'fixation.txt')

    archivos_descargados = {
        "timestamps": False,
        "fixation": False
    }

    for blob in bucket.list_blobs(prefix=folder):
        print(f"Archivo encontrado: {blob.name}")
        if blob.name.endswith('timestamps.txt'):
            blob.download_to_filename(estimulos_path)
            archivos_descargados["timestamps"] = True
            print(f"Archivo descargado: {estimulos_path}")
        elif blob.name.endswith('fixation.txt'):
            blob.download_to_filename(fijaciones_path)
            archivos_descargados["fixation"] = True
            print(f"Archivo descargado: {fijaciones_path}")

    # Verificar si ambos archivos se descargaron correctamente
    if not archivos_descargados["timestamps"] or not archivos_descargados["fixation"]:
        return "<h1>Error: No se pudieron descargar ambos archivos necesarios desde Firebase.</h1>"

    # Procesar los datos descargados
    estimulos = leer_estimulos(estimulos_path, range(7))  # Ajustar líneas a omitir
    fijaciones = leer_fijaciones(fijaciones_path, [0])  # Ajustar líneas a omitir
    comparar_tiempos(fijaciones, estimulos, RESULT_FILE)
    print(estimulos)
    print(fijaciones)

    result_blob = bucket.blob(f"{folder}resultados.txt")
    result_blob.upload_from_filename(RESULT_FILE)
    print(f"Archivo de resultados subido a Firebase en la carpeta {folder}")
    
    # Leer el archivo de resultados desde Firebase
    with open(RESULT_FILE, 'r') as file:
        lineas = file.readlines()

    # Extraer las posiciones del archivo (suponiendo que las posiciones estén en cada línea del archivo)
    positions = []
    for linea in lineas:
        try:
            # Si las posiciones están en formato "x", conviértelas a enteros.
            x = int(linea.strip())  # Asumimos que cada línea contiene un valor de X
            positions.append(x)  # Guardar las posiciones X en una lista
        except ValueError:
            continue  # Ignorar líneas con formato incorrecto
        
    # Accede al documento del último paciente
    doc_ref = db.collection('lastpatient').document('lastpatient')
    doc = doc_ref.get()

    # Verifica si el documento existe y luego obtiene los datos
    if doc.exists:
            patient_data = doc.to_dict()
            ID = patient_data.get("id")
            FDN = patient_data.get("dob")
            sexo = patient_data.get("gender")
            edad = patient_data.get("age")
            estimulo = patient_data.get("examConfig", {}).get("stimulusSize")
            patron = patient_data.get("examConfig", {}).get("pattern")
            ojo = patient_data.get("examConfig", {}).get("eye")
            diametro = 4  # Valor fijo, ajusta según tu lógica
            apellido = patient_data.get("lastName")

    bucket = storage.bucket()
    if not bucket:
        raise Exception("No se pudo acceder al bucket de almacenamiento.")

    # Descarga el archivo input_data.txt
    file_name = f'Postprocesamiento/{ID}_{apellido}.txt'
    #file_name = 'Postprocesamiento/62_Fjw (4).txt'
    blob = bucket.blob(file_name)
    input_data = blob.download_as_text()

    # Procesar el contenido del archivo para extraer los números en un array llamado data_array
    data_array = [int(value.strip()) for value in input_data.split(",")]  # Ajusta el separador según el formato del archivo

    # Reemplazar valores en las posiciones indicadas en positions por 0
    for x in positions:
        if 0 <= x < len(data_array):  # Verificar que la posición X esté dentro de los límites del array
            data_array[x] = 0  # Reemplazar el valor en la posición X con 0

    # Generar contenido para el archivo de resultados
    file_content = f"""Datos del Paciente:
    Nombre: {ID} {apellido}
    Fecha de Nacimiento: {FDN}, Edad: {edad}, Género: {sexo}

    Configuración del Examen:
    Fondo: Blanco 31.5 asb, Ojo: {ojo}, Patrón: {patron}, Color estímulo: Blanco, Tamaño estímulo: {estimulo}, Diámetro: {diametro}, Estrategia: SITA Estándar

    Intensidades Observadas:
    """
    # Agregar las intensidades observadas
    for index, value in enumerate(data_array):
        file_content += f"Posición {index + 1}: {value}\n"

    # Guardar el archivo en formato .txt
    output_file_path = os.path.join(POST_FOLDER, f'{ID}_{apellido}_processed.txt')  # Ruta para guardar el archivo localmente
    with open(output_file_path, 'w') as output_file:
        # Escribir contenido formateado en el archivo
        output_file.write(file_content)

    print(f"Archivo formateado guardado en {output_file_path}")

    # Subir el archivo al bucket de Firebase Storage
    output_blob = bucket.blob(f'results/{ID}_{apellido}_processed.txt')  # Nombre del archivo en el bucket
    output_blob.upload_from_filename(output_file_path)

    print(f"Archivo actualizado subido a Firebase Storage en results/{ID}_{apellido}_updated.txt")

    return '''
    <h1>Archivos procesados correctamente</h1>
    <p>Los archivos de Firebase han sido procesados.</p>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

