import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import pickle
from mtcnn.mtcnn import MTCNN
import numpy as np
import mediapipe as mp

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
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# ----------- REGISTRO FACIAL EN VENTANA OPENCV CON MEDIAPIPE -----------
def registrar_facial():
    nombre = entry_usuario_facial.get()
    if not nombre:
        messagebox.showerror("Error", "Introduce un nombre de usuario.")
        return

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    registrado = False

    while True:
        ret, frame = cap.read()
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
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    faces.append((x, y, w, h))
        cv2.putText(frame, f"Personas detectadas: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Registro facial (presiona 's' para capturar, ESC para salir)", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC para salir
            break
        if key == ord('s') and faces:
            x, y, w, h = faces[0]
            # Asegura que los valores estén dentro del frame
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)   
            h = max(1, h)
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                rostro = frame[y:y+h, x:x+w]
                path_img = f"{DATA_DIR}/{nombre}.jpg"
                cv2.imwrite(path_img, rostro)
                registrado = True
            else:
                messagebox.showerror("Error", "No se pudo recortar el rostro. Intenta de nuevo.")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    if registrado:
        messagebox.showinfo("Éxito", "Rostro registrado correctamente.")
        entry_usuario_facial.delete(0, tk.END)
    # Solo muestra advertencia si el usuario intentó capturar y no se detectó rostro
    # Si solo salió con ESC, no muestra nada

# ----------- LOGIN FACIAL EN VENTANA OPENCV CON MEDIAPIPE -----------
def login_facial():
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    login_exitoso = False
    mejor_usuario = None

    while True:
        ret, frame = cap.read()
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
            mejor_score = -1
            for archivo in os.listdir(DATA_DIR):
                if archivo.endswith(".jpg"):
                    score = comparar_rostros(os.path.join(DATA_DIR, archivo), rostro_login)
                    if score > mejor_score:
                        mejor_score = score
                        mejor_usuario = archivo.replace(".jpg", "")
            if mejor_score > 0.7:
                login_exitoso = True
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    if login_exitoso:
        messagebox.showinfo("Bienvenido", f"Login facial exitoso. Hola, {mejor_usuario}!")
    else:
        messagebox.showerror("Error", "No se encontró coincidencia.")

# ----------- REGISTRO Y LOGIN MANUAL (Tkinter) -----------
def registrar_manual():
    usuario = entry_usuario_manual.get()
    contra = entry_contra_manual.get()
    if not usuario or not contra:
        messagebox.showerror("Error", "Completa usuario y contraseña.")
        return
    guardar_usuario_manual(usuario, contra)
    messagebox.showinfo("Éxito", "Usuario registrado manualmente.")
    entry_usuario_manual.delete(0, tk.END)
    entry_contra_manual.delete(0, tk.END)

def login_manual():
    usuario = entry_usuario_login.get()
    contra = entry_contra_login.get()
    if verificar_usuario_manual(usuario, contra):
        messagebox.showinfo("Bienvenido", f"Login manual exitoso. Hola, {usuario}!")
    else:
        messagebox.showerror("Error", "Usuario o contraseña incorrectos.")

# ----------- INTERFAZ PRINCIPAL -----------
root = tk.Tk()
root.title("Sistema de Usuarios: Registro y Login Facial/Manual")
root.geometry("600x500")
root.configure(bg="#222222")

tabControl = ttk.Notebook(root)
tab_registro = ttk.Frame(tabControl)
tab_login = ttk.Frame(tabControl)
tabControl.add(tab_registro, text='Registro')
tabControl.add(tab_login, text='Login')
tabControl.pack(expand=1, fill="both")

# Registro facial
tk.Label(tab_registro, text="Registro facial (solo usuario):", font=("Arial", 10, "bold")).pack(pady=5)
entry_usuario_facial = tk.Entry(tab_registro)
entry_usuario_facial.pack(pady=2)
btn_registrar_facial = tk.Button(tab_registro, text="Registrar y Capturar Foto", command=registrar_facial)
btn_registrar_facial.pack(pady=5)

# Registro manual
tk.Label(tab_registro, text="Registro manual (usuario y contraseña):", font=("Arial", 10, "bold")).pack(pady=10)
tk.Label(tab_registro, text="Usuario:").pack()
entry_usuario_manual = tk.Entry(tab_registro)
entry_usuario_manual.pack()
tk.Label(tab_registro, text="Contraseña:").pack()
entry_contra_manual = tk.Entry(tab_registro, show="*")
entry_contra_manual.pack()
btn_registrar_manual = tk.Button(tab_registro, text="Registrar Manual", command=registrar_manual)
btn_registrar_manual.pack(pady=5)

# Login facial
tk.Label(tab_login, text="Login facial:", font=("Arial", 10, "bold")).pack(pady=5)
btn_login_facial = tk.Button(tab_login, text="Login Facial", command=login_facial)
btn_login_facial.pack(pady=5)

# Login manual
tk.Label(tab_login, text="Login manual:", font=("Arial", 10, "bold")).pack(pady=10)
tk.Label(tab_login, text="Usuario:").pack()
entry_usuario_login = tk.Entry(tab_login)
entry_usuario_login.pack()
tk.Label(tab_login, text="Contraseña:").pack()
entry_contra_login = tk.Entry(tab_login, show="*")
entry_contra_login.pack()
btn_login_manual = tk.Button(tab_login, text="Login Manual", command=login_manual)
btn_login_manual.pack(pady=5)

root.mainloop()
