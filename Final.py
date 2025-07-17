import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import pickle

# Carpeta donde se guardarán los rostros registrados
DATA_DIR = "rostros_registrados"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def mostrar_imagen(path, label):
    img = Image.open(path)
    img = img.resize((150, 150))
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

def capturar_y_mostrar(nombre, label):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{DATA_DIR}/{nombre}.jpg", frame)
        mostrar_imagen(f"{DATA_DIR}/{nombre}.jpg", label)
    cap.release()
    cv2.destroyAllWindows()

def registrar_usuario():
    nombre = entry_usuario.get()
    if not nombre:
        messagebox.showerror("Error", "Introduce un nombre de usuario.")
        return
    capturar_y_mostrar(nombre, label_foto_registro)
    # Codificar y guardar el rostro
    imagen = face_recognition.load_image_file(f"{DATA_DIR}/{nombre}.jpg")
    encoding = face_recognition.face_encodings(imagen)
    if encoding:
        with open(f"{DATA_DIR}/{nombre}.pkl", "wb") as f:
            pickle.dump(encoding[0], f)
        messagebox.showinfo("Éxito", "Usuario registrado correctamente.")
        entry_usuario.delete(0, tk.END)
    else:
        messagebox.showerror("Error", "No se detectó ningún rostro. Intenta de nuevo.")

def login_usuario():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("login_temp.jpg", frame)
        mostrar_imagen("login_temp.jpg", label_foto_login)
    cap.release()
    cv2.destroyAllWindows()
    imagen_login = face_recognition.load_image_file("login_temp.jpg")
    encoding_login = face_recognition.face_encodings(imagen_login)
    if not encoding_login:
        messagebox.showerror("Error", "No se detectó ningún rostro.")
        return
    encoding_login = encoding_login[0]
    # Comparar con los usuarios registrados
    for archivo in os.listdir(DATA_DIR):
        if archivo.endswith(".pkl"):
            with open(f"{DATA_DIR}/{archivo}", "rb") as f:
                encoding_registrado = pickle.load(f)
            resultado = face_recognition.compare_faces([encoding_registrado], encoding_login)
            if resultado[0]:
                usuario = archivo.replace(".pkl", "")
                messagebox.showinfo("Bienvenido", f"Login exitoso. Hola, {usuario}!")
                return
    messagebox.showerror("Error", "No se encontró coincidencia.")

# Interfaz gráfica mejorada
root = tk.Tk()
root.title("Registro y Login Facial")
root.geometry("450x400")

tabControl = ttk.Notebook(root)
tab_registro = ttk.Frame(tabControl)
tab_login = ttk.Frame(tabControl)
tabControl.add(tab_registro, text='Registro')
tabControl.add(tab_login, text='Login')
tabControl.pack(expand=1, fill="both")

# Registro
tk.Label(tab_registro, text="Usuario:").pack(pady=5)
entry_usuario = tk.Entry(tab_registro)
entry_usuario.pack(pady=5)
btn_registrar = tk.Button(tab_registro, text="Registrar y Capturar Foto", command=registrar_usuario)
btn_registrar.pack(pady=10)
label_foto_registro = tk.Label(tab_registro)
label_foto_registro.pack(pady=10)

# Login
btn_login = tk.Button(tab_login, text="Login Facial", command=login_usuario)
btn_login.pack(pady=10)
label_foto_login = tk.Label(tab_login)
label_foto_login.pack(pady=10)

root.mainloop()

