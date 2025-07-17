import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import pickle
from mtcnn.mtcnn import MTCNN
import numpy as np

# -------------------- CONFIGURACIÓN DE CARPETAS Y ARCHIVOS --------------------
DATA_DIR = "rostros_registrados"
USERS_FILE = "usuarios.pkl"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# -------------------- FUNCIONES DE UTILIDAD --------------------
def guardar_usuario_manual(usuario, contra):
    """Guarda usuario y contraseña en archivo local."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as f:
            usuarios = pickle.load(f)
    else:
        usuarios = {}
    usuarios[usuario] = contra
    with open(USERS_FILE, "wb") as f:
        pickle.dump(usuarios, f)

def verificar_usuario_manual(usuario, contra):
    """Verifica usuario y contraseña."""
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, "rb") as f:
        usuarios = pickle.load(f)
    return usuarios.get(usuario) == contra

def mostrar_imagen_cv2(frame, label):
    """Convierte un frame de OpenCV a imagen de Tkinter y la muestra en un label."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

def comparar_rostros(img1_path, img2):
    """
    Compara dos imágenes de rostro usando histogramas (simple, no biométrico).
    img1_path: ruta de la imagen registrada
    img2: imagen capturada (numpy array)
    """
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# -------------------- REGISTRO FACIAL CON CÁMARA ACTIVA Y CONTEO --------------------
def registrar_facial():
    nombre = entry_usuario_facial.get()
    if not nombre:
        messagebox.showerror("Error", "Introduce un nombre de usuario.")
        return

    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    rostro_guardado = False

    ventana = tk.Toplevel(root)
    ventana.title("Registro facial en vivo")
    ventana.geometry("500x350")
    label_video = tk.Label(ventana)
    label_video.pack()
    label_conteo = tk.Label(ventana, text="Personas detectadas: 0", font=("Arial", 12))
    label_conteo.place(relx=1.0, rely=1.0, anchor="se")  # Esquina inferior derecha

    def actualizar():
        nonlocal rostro_guardado
        ret, frame = cap.read()
        if not ret:
            ventana.after(10, actualizar)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        # Dibuja rectángulos y cuenta personas
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        label_conteo.config(text=f"Personas detectadas: {len(faces)}")
        mostrar_imagen_cv2(frame, label_video)
        if not rostro_guardado:
            ventana.after(10, actualizar)

    def capturar_rostro():
        nonlocal rostro_guardado
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            rostro = frame[y:y+h, x:x+w]
            path_img = f"{DATA_DIR}/{nombre}.jpg"
            cv2.imwrite(path_img, rostro)
            messagebox.showinfo("Éxito", "Rostro registrado correctamente.")
            rostro_guardado = True
            cap.release()
            ventana.destroy()
            entry_usuario_facial.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "No se detectó ningún rostro. Intenta de nuevo.")

    btn_capturar = tk.Button(ventana, text="Capturar rostro", command=capturar_rostro)
    btn_capturar.pack(pady=10)
    actualizar()

# -------------------- LOGIN FACIAL CON CÁMARA ACTIVA Y CONTEO --------------------
def login_facial():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    login_exitoso = False

    ventana = tk.Toplevel(root)
    ventana.title("Login facial en vivo")
    ventana.geometry("500x350")
    label_video = tk.Label(ventana)
    label_video.pack()
    label_conteo = tk.Label(ventana, text="Personas detectadas: 0", font=("Arial", 12))
    label_conteo.place(relx=1.0, rely=1.0, anchor="se")  # Esquina inferior derecha

    def actualizar():
        if login_exitoso:
            return
        ret, frame = cap.read()
        if not ret:
            ventana.after(10, actualizar)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        # Dibuja rectángulos y cuenta personas
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        label_conteo.config(text=f"Personas detectadas: {len(faces)}")
        mostrar_imagen_cv2(frame, label_video)
        ventana.after(10, actualizar)

    def intentar_login():
        nonlocal login_exitoso
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if not faces:
            messagebox.showerror("Error", "No se detectó ningún rostro.")
            return
        x, y, w, h = faces[0]['box']
        rostro_login = frame[y:y+h, x:x+w]
        mejor_score = -1
        mejor_usuario = None
        # Compara con todos los rostros registrados
        for archivo in os.listdir(DATA_DIR):
            if archivo.endswith(".jpg"):
                score = comparar_rostros(os.path.join(DATA_DIR, archivo), rostro_login)
                if score > mejor_score:
                    mejor_score = score
                    mejor_usuario = archivo.replace(".jpg", "")
        if mejor_score > 0.7:  # Umbral simple, puedes ajustarlo
            messagebox.showinfo("Bienvenido", f"Login facial exitoso. Hola, {mejor_usuario}!")
            login_exitoso = True
            cap.release()
            ventana.destroy()
        else:
            messagebox.showerror("Error", "No se encontró coincidencia.")

    btn_login = tk.Button(ventana, text="Intentar login facial", command=intentar_login)
    btn_login.pack(pady=10)
    actualizar()

# -------------------- REGISTRO MANUAL --------------------
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

# -------------------- LOGIN MANUAL --------------------
def login_manual():
    usuario = entry_usuario_login.get()
    contra = entry_contra_login.get()
    if verificar_usuario_manual(usuario, contra):
        messagebox.showinfo("Bienvenido", f"Login manual exitoso. Hola, {usuario}!")
    else:
        messagebox.showerror("Error", "Usuario o contraseña incorrectos.")

# -------------------- INTERFAZ GRÁFICA PRINCIPAL --------------------
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

# -------------------- REGISTRO --------------------
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

# -------------------- LOGIN --------------------
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

# -------------------- EXPLICACIÓN GENERAL --------------------
# - El registro facial y login facial usan una cámara activa y muestran el conteo de personas detectadas en tiempo real.
# - El usuario debe presionar el botón "Capturar rostro" o "Intentar login facial" para guardar o comparar el rostro detectado.
# - El registro y login manual funcionan con usuario y contraseña, guardados en un archivo local.
# - Los rostros se guardan en la carpeta 'rostros_registrados' y los usuarios manuales en 'usuarios.pkl'.
# - El sistema es modular y puedes modificar los umbrales o la forma de comparar rostros según tus necesidades.