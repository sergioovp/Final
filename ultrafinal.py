import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import pickle

import mediapipe as mp
import face_recognition

logofisi = "logo-fisi.ico"  



DATA_DIR = "rostros_registrados"
USERS_FILE = "usuarios.pkl"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def guardar_usuario_manual(usuario, contra):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as f:
            usuarios = pickle.load(f)
    else:
        usuarios = {}
    usuarios[usuario] = contra
    with open(USERS_FILE, "wb") as f:
        pickle.dump(usuarios, f)

def verificar_usuario_manual(usuario, contra):
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, "rb") as f:
        usuarios = pickle.load(f)
    return usuarios.get(usuario) == contra

def comparar_rostros(img1_path, img2):
    """
    Compara dos imágenes de rostro usando embeddings de face_recognition/dlib.
    img1_path: ruta de la imagen registrada
    img2: imagen capturada (numpy array BGR)
    Retorna True si son la misma persona, False si no.
    """
    # Carga la imagen registrada
    img1 = face_recognition.load_image_file(img1_path)
    # Convierte la imagen capturada a RGB
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Obtiene los vectores de características (embeddings)
    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2_rgb)
    if len(encodings1) == 0 or len(encodings2) == 0:
        return False  # No se detectó rostro en alguna imagen
    # Compara los vectores
    resultado = face_recognition.compare_faces([encodings1[0]], encodings2[0], tolerance=0.5)
    return resultado[0]

def comparar_rostros_orb(img1_path, img2):
    """
    Compara dos imágenes de rostro usando ORB.
    img1_path: ruta de la imagen registrada
    img2: imagen capturada (numpy array BGR)
    Retorna True si la similitud es mayor a un umbral.
    """
    img1 = cv2.imread(img1_path)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    similitud = orb_sim(img1_gray, img2_gray)
    return similitud > 0.3  # Puedes ajustar el umbral según tus pruebas

# ----------- REGISTRO FACIAL EN VENTANA OPENCV CON MEDIAPIPE -----------
def registrar_facial():
    nombre = entradadelusuariofacial.get()
    if not nombre:
        messagebox.showerror("Recuerda", "Introduce un nombre de usuario.")
        return
    captura = cv2.VideoCapture(0)
    caraenmediapipe = mp.solutions.face_detection
    face_detection = caraenmediapipe.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    registrado = False

    while True:
        ret, frame = captura.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                # Asegura que los valores estén dentro del frame
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)
                if y+h <= ih and x+w <= iw:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,105,0), 2)
                    faces.append((x, y, w, h))
        cv2.putText(frame, f"Personas detectadas: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Registro facial (presiona 's' para capturar, ESC para salir)", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC para salir
            break
        if key == ord('s'):
            if faces:
                x, y, w, h = faces[0]
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)
                if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                    rostro = frame[y:y+h, x:x+w]
                    path_img = f"{DATA_DIR}/{nombre}.jpg"
                    cv2.imwrite(path_img, rostro)
                    registrado = True
                    messagebox.showinfo("Éxito", "Rostro registrado correctamente.")
                    entradadelusuariofacial.delete(0, tk.END)
                else:
                    messagebox.showerror("Error", "No se pudo recortar el rostro. Intenta de nuevo.")
            else:
                messagebox.showerror("Error", "No se detectó ningún rostro al capturar. Intenta de nuevo.")
            break

    captura.release()
    cv2.destroyAllWindows()
    face_detection.close()

# ----------- LOGIN FACIAL EN VENTANA OPENCV CON MEDIAPIPE -----------
def login_facial():
    captura = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    login_exitoso = False
    mejor_usuario = None

    while True:
        ret, frame = captura.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                faces.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"Personas detectadas: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Login facial (presiona 's' para intentar login, ESC para salir)", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC para salir
            break
        if key == ord('s') and faces:
            x, y, w, h = faces[0]
            rostro_login = frame[y:y+h, x:x+w]
            mejor_score = -1  # <-- Definir la variable antes de usarla
            mejor_usuario = None
            for archivo in os.listdir(DATA_DIR):
                if archivo.endswith(".jpg"):
                    score = 0
                    try:
                        score = comparar_rostros(os.path.join(DATA_DIR, archivo), rostro_login)
                    except Exception:
                        continue
                    # Si comparar_rostros retorna True/False (face_recognition), usar así:
                    if isinstance(score, bool):
                        if score:
                            mejor_usuario = archivo.replace(".jpg", "")
                            login_exitoso = True
                            break
                    else:
                        # Si retorna un valor de similitud, usar el umbral
                        if score > mejor_score:
                            mejor_score = score
                            mejor_usuario = archivo.replace(".jpg", "")
            # Si usar face_recognition, login_exitoso ya se habrá puesto en True
            # Si usar similitud, verifica el umbral
            if not login_exitoso and mejor_score > 0.7:
                login_exitoso = True
            break

    captura.release()
    cv2.destroyAllWindows()
    face_detection.close()
    if login_exitoso:
        messagebox.showinfo("Bienvenido", f"Login facial exitoso. Hola, {mejor_usuario}!")
    else:
        messagebox.showerror("Error", "No se encontró coincidencia.")
    
# ----------- REGISTRO Y LOGIN MANUAL (Tkinter) -----------
def registrar_manual():
    usuario = entradadelusuariomanual.get()
    contra = entradacontramanual.get()
    if not usuario or not contra:
        messagebox.showerror("Error", "Coloca un usuario.")
        return
    guardar_usuario_manual(usuario, contra)
    messagebox.showinfo("Éxito", "Usuario registrado manualmente.")
    entradadelusuariomanual.delete(0, tk.END)
    entradacontramanual.delete(0, tk.END)

def login_manual():
    usuario = entradausuariologin.get()
    contra = entradacontralogin.get()
    if verificar_usuario_manual(usuario, contra):
        messagebox.showinfo("Bienvenido", f"Login manual exitoso. Hola, {usuario}!")
    else:
        messagebox.showerror("Error", "Usuario o contraseña incorrectos.")

def orb_sim(img1, img2):
    """
    Compara dos imágenes usando ORB y retorna el porcentaje de similitud.
    img1, img2: imágenes en escala de grises (numpy arrays)
    """
    orb = cv2.ORB_create()
    kpa, descr_a = orb.detectAndCompute(img1, None)
    kpb, descr_b = orb.detectAndCompute(img2, None)
    if descr_a is None or descr_b is None:
        return 0
    comp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = comp.match(descr_a, descr_b)
    regiones_similares = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(regiones_similares) / len(matches)

def centrar_ventana(ventana, ancho, alto):
    #obtener dimensiones de la pantalla
    anchopantalla=ventana.winfo_screenwidth()
    alturapantalla=ventana.winfo_screenheight()
    #calcular coordenadas para centrar
    x=(anchopantalla// 2)-(ancho// 2)
    y=(alturapantalla// 2)-(alto// 2)
    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")
# -----------###INTERFAZ PRINCIPAL###-----------
pantallainicial = tk.Tk()
pantallainicial.title("Proyecto Grupo 1: GatOs")
pantallainicial.configure(bg="#FFFFFF")
pantallainicial.iconbitmap(logofisi)  
tabul = ttk.Notebook(pantallainicial)
tab_registro = ttk.Frame(tabul)
tab_login = ttk.Frame(tabul)
tab_creditos = ttk.Frame(tabul)    
tabul.add(tab_registro, text='Registro')
tabul.add(tab_login, text='Login')
tabul.add(tab_creditos, text='Creditos')
tabul.pack(expand=1, fill="both")
ancho_ventana = 400
alto_ventana = 600
centrar_ventana(pantallainicial, ancho_ventana, alto_ventana)
#CREDITOS:
tk.Label(tab_creditos, text="Nombres de los integrantes:", font=("Helvetica", 20, "bold")).pack(pady=5)
tk.Label(tab_creditos, text="Morales Japa  Marlon", font=("Helvetica", 10, "bold")).pack(pady=5)
tk.Label(tab_creditos, text="Valencia Pastrana Sergio Daniel", font=("Helvetica", 10, "bold")).pack(pady=5)
tk.Label(tab_creditos, text="Janampa Mayta  Yonatan David", font=("Helvetica", 10, "bold")).pack(pady=5)
tk.Label(tab_creditos, text="Castillo Flores  Manuel Joaquin", font=("Helvetica", 10, "bold")).pack(pady=5)
#REGISTRO FACIAL
tk.Label(tab_registro, text="Grupo 1: GatOs", font=("Helvetica", 20, "bold")).pack(pady=5)
tk.Label(tab_registro, text="Registro facial (solo usuario):", font=("Helvetica", 10, "bold")).pack(pady=5)
entradadelusuariofacial = tk.Entry(tab_registro)
entradadelusuariofacial.pack(pady=2)
botonregistrofacial = tk.Button(tab_registro, text="Registrar y Capturar Foto", command=registrar_facial)
botonregistrofacial.pack(pady=5)
 
#REGISTRO manual
tk.Label(tab_registro, text="Registro manual (usuario y contraseña):", font=("Arial", 10, "bold")).pack(pady=10)
tk.Label(tab_registro, text="Usuario:").pack()
entradadelusuariomanual = tk.Entry(tab_registro)
entradadelusuariomanual.pack()
tk.Label(tab_registro, text="Contraseña:").pack()
entradacontramanual = tk.Entry(tab_registro, show="*")
entradacontramanual.pack()
botonregistromanual = tk.Button(tab_registro, text="Registrar Manual", command=registrar_manual)
botonregistromanual.pack(pady=5)

# Login facial
tk.Label(tab_login, text="Login facial:", font=("Arial", 12, "bold")).pack(pady=5)
botonfaciallogin = tk.Button(tab_login, text="Login Facial", command=login_facial)
botonfaciallogin.pack(pady=5)

# Login manual
tk.Label(tab_login, text="Login manual:", font=("Arial", 12, "bold")).pack(pady=10)
tk.Label(tab_login, text="Usuario:").pack()
entradausuariologin = tk.Entry(tab_login)
entradausuariologin.pack()
tk.Label(tab_login, text="Contraseña:").pack()
entradacontralogin = tk.Entry(tab_login, show="*")
entradacontralogin.pack()
botonloginmanual = tk.Button(tab_login, text="Login Manual", command=login_manual)
botonloginmanual.pack(pady=5)
ancho_ventana = 400
alto_ventana = 600
centrar_ventana(pantallainicial, ancho_ventana, alto_ventana)       
pantallainicial.mainloop()
