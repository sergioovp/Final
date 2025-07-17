#--------------------------------------Importamos librerias--------------------------------------------

from tkinter import *
from tkinter import messagebox
import os
import cv2
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN
import numpy as np

# -------------------- CONFIGURACIÓN DE CARPETAS Y ARCHIVOS --------------------
DATA_DIR = "rostros_registrados"
USERS_FILE = "usuarios.txt"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# -------------------- FUNCIONES DE UTILIDAD --------------------
def guardar_usuario_manual(usuario, contra):
    """Guarda usuario y contraseña en un archivo de texto."""
    with open(USERS_FILE, "a") as archivo:
        archivo.write(f"{usuario},{contra}\n")

def verificar_usuario_manual(usuario, contra):
    """Verifica usuario y contraseña leyendo el archivo de texto."""
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, "r") as archivo:
        for linea in archivo:
            usuario_arch, contra_arch = linea.strip().split(",", 1)
            if usuario == usuario_arch and contra == contra_arch:
                return True
    return False

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
    nombre = usuario.get().strip()
    if not nombre:
        messagebox.showerror("Error", "Introduce un nombre de usuario.")
        return

    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    rostro_guardado = False

    ventana = Toplevel(pantalla1)
    ventana.title("Registro facial en vivo")
    ventana.geometry("500x400")
    label_video = Label(ventana)
    label_video.pack()
    label_conteo = Label(ventana, text="Personas detectadas: 0", font=("Arial", 12))
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
        # Mostrar la imagen en la ventana de Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_video.config(image=img_tk)
        label_video.image = img_tk
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
            path_img = os.path.join(DATA_DIR, f"{nombre}.jpg")
            cv2.imwrite(path_img, rostro)
            messagebox.showinfo("Éxito", "Rostro registrado correctamente.")
            rostro_guardado = True
            cap.release()
            ventana.destroy()
        else:
            messagebox.showerror("Error", "No se detectó ningún rostro. Intenta de nuevo.")

    Button(ventana, text="Capturar rostro", command=capturar_rostro).pack(pady=10)
    actualizar()

# -------------------- LOGIN FACIAL CON CÁMARA ACTIVA Y CONTEO --------------------
def login_facial():
    nombre = verificacion_usuario.get().strip()
    if not nombre:
        messagebox.showerror("Error", "Introduce tu usuario para login facial.")
        return

    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    login_exitoso = False

    ventana = Toplevel(pantalla2)
    ventana.title("Login facial en vivo")
    ventana.geometry("500x400")
    label_video = Label(ventana)
    label_video.pack()
    label_conteo = Label(ventana, text="Personas detectadas: 0", font=("Arial", 12))
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
        # Mostrar la imagen en la ventana de Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_video.config(image=img_tk)
        label_video.image = img_tk
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
        path_registrado = os.path.join(DATA_DIR, f"{nombre}.jpg")
        if os.path.exists(path_registrado):
            score = comparar_rostros(path_registrado, rostro_login)
            if score > 0.7:  # Umbral simple, puedes ajustarlo
                messagebox.showinfo("Bienvenido", f"Login facial exitoso. Hola, {nombre}!")
                login_exitoso = True
                cap.release()
                ventana.destroy()
            else:
                messagebox.showerror("Error", "El rostro no coincide con el registrado.")
        else:
            messagebox.showerror("Error", "No hay registro facial para este usuario.")

    Button(ventana, text="Intentar login facial", command=intentar_login).pack(pady=10)
    actualizar()

# -------------------- REGISTRO MANUAL --------------------
def registrar_usuario():
    usuario_info = usuario.get().strip()
    contra_info = contra.get().strip()
    if not usuario_info or not contra_info:
        Label(pantalla1, text="Debes ingresar usuario y contraseña.", fg="red", font=("Calibri",11)).pack()
        return
    guardar_usuario_manual(usuario_info, contra_info)
    usuario_entrada.delete(0, END)
    contra_entrada.delete(0, END)
    Label(pantalla1, text="Registro Convencional Exitoso", fg="green", font=("Calibri",11)).pack()

# -------------------- LOGIN MANUAL --------------------
def verificacion_login():
    log_usuario = verificacion_usuario.get().strip()
    log_contra = verificacion_contra.get().strip()
    usuario_entrada2.delete(0, END)
    contra_entrada2.delete(0, END)
    if verificar_usuario_manual(log_usuario, log_contra):
        Label(pantalla2, text="Inicio de Sesion Exitoso", fg="green", font=("Calibri",11)).pack()
    else:
        Label(pantalla2, text="Usuario o contraseña incorrectos", fg="red", font=("Calibri",11)).pack()

#------------------------Crearemos una funcion para asignar al boton registro --------------------------------
def registro():
    global usuario
    global contra
    global usuario_entrada
    global contra_entrada
    global pantalla1
    pantalla1 = Toplevel(pantalla)
    pantalla1.title("Registro")
    pantalla1.geometry("300x250")
    usuario = StringVar()
    contra = StringVar()
    Label(pantalla1, text="Registro facial: debe de asignar un usuario:").pack()
    Label(pantalla1, text="Registro tradicional: debe asignar usuario y contraseña:").pack()
    Label(pantalla1, text="Usuario * ").pack()
    usuario_entrada = Entry(pantalla1, textvariable=usuario)
    usuario_entrada.pack()
    Label(pantalla1, text="Contraseña * ").pack()
    contra_entrada = Entry(pantalla1, textvariable=contra, show="*")
    contra_entrada.pack()
    Button(pantalla1, text="Registro Tradicional", width=15, height=1, command=registrar_usuario).pack(pady=5)
    Button(pantalla1, text="Registro Facial", width=15, height=1, command=registrar_facial).pack(pady=5)

#------------------------- Funcion para el Login ------------------------------------------------
def login():
    global pantalla2
    global verificacion_usuario
    global verificacion_contra
    global usuario_entrada2
    global contra_entrada2
    pantalla2 = Toplevel(pantalla)
    pantalla2.title("Login")
    pantalla2.geometry("300x250")
    verificacion_usuario = StringVar()
    verificacion_contra = StringVar()
    Label(pantalla2, text="Login facial: debe de asignar un usuario:").pack()
    Label(pantalla2, text="Login tradicional: debe asignar usuario y contraseña:").pack()
    Label(pantalla2, text="Usuario * ").pack()
    usuario_entrada2 = Entry(pantalla2, textvariable=verificacion_usuario)
    usuario_entrada2.pack()
    Label(pantalla2, text="Contraseña * ").pack()
    contra_entrada2 = Entry(pantalla2, textvariable=verificacion_contra, show="*")
    contra_entrada2.pack()
    Button(pantalla2, text="Inicio de Sesion Tradicional", width=20, height=1, command=verificacion_login).pack(pady=5)
    Button(pantalla2, text="Inicio de Sesion Facial", width=20, height=1, command=login_facial).pack(pady=5)

#------------------------- Funcion de nuestra pantalla principal ------------------------------------------------
def pantalla_principal():
    global pantalla
    pantalla = Tk()
    pantalla.geometry("300x250")
    pantalla.title("Aprende e Ingenia")
    Label(text="Login Inteligente", bg="gray", width="300", height="2", font=("Verdana", 13)).pack()
    Label(text="").pack()
    Button(text="Iniciar Sesion", height="2", width="30", command=login).pack()
    Label(text="").pack()
    Button(text="Registro", height="2", width="30", command=registro).pack()
    pantalla.mainloop()

pantalla_principal()

# -------------------- EXPLICACIÓN GENERAL --------------------
# - El registro facial y login facial usan una cámara activa y muestran el conteo de personas detectadas en tiempo real.
# - El usuario debe presionar el botón "Capturar rostro" o "Intentar login facial" para guardar o comparar el rostro detectado.
# - El registro y login manual funcionan con usuario y contraseña, guardados en un archivo local.
# - Los rostros se guardan en la carpeta 'rostros_registrados' y los usuarios manuales en 'usuarios.txt'.
# - El sistema es modular y puedes modificar los umbrales o la forma de comparar rostros según tus necesidades.
