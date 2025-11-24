import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

# ---------- CONFIG GLOBALE ----------

DATASET_DIR = "dataset"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "trainer.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Couleurs globales (style)
PRIMARY = "#4E7CFF"
DARK_BG = "#111318"
DARK_BG_ALT = "#181b22"
DARK_FG = "#FFFFFF"
LIGHT_BG = "#FFFFFF"
LIGHT_FG = "#000000"

# ---------- UTILITAIRES ----------

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

def load_labels(labels_path):
    label_map = {}
    if not os.path.exists(labels_path):
        return label_map
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, name = line.split(";", 1)
            label_map[int(id_str)] = name
    return label_map

def train_model(status_callback=None):
    """
    Entraîne le modèle LBPH à partir des images du dossier dataset/.
    """
    if status_callback:
        status_callback("Chargement du dataset...")

    if not os.path.isdir(DATASET_DIR):
        if status_callback:
            status_callback("Aucun dossier dataset/ trouvé.")
        return False

    people = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    if not people:
        if status_callback:
            status_callback("Aucun visage dans dataset/. Ajoute d'abord des visages.")
        return False

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        label_map[current_id] = person

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_id)

        current_id += 1

    if not faces:
        if status_callback:
            status_callback("Aucune image valide trouvée dans dataset/.")
        return False

    faces = np.array(faces, dtype="uint8")
    labels = np.array(labels, dtype="int32")

    if status_callback:
        status_callback("Entraînement du modèle LBPH...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for id_val, name in label_map.items():
            f.write(f"{id_val};{name}\n")

    if status_callback:
        status_callback("Entraînement terminé ✔")
    return True

# ---------- CLASSES D'ÉCRANS ----------

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller

        title = tk.Label(
            self,
            text="E-Face",
            font=("Segoe UI", 28, "bold"),
            bg=DARK_BG,
            fg=PRIMARY
        )
        subtitle = tk.Label(
            self,
            text="Reconnaissance faciale locale",
            font=("Segoe UI", 13),
            bg=DARK_BG,
            fg="#888888"
        )
        title.pack(pady=(40, 5))
        subtitle.pack(pady=(0, 30))

        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(pady=10)

        def make_button(text, command):
            b = tk.Button(
                btn_frame,
                text=text,
                font=("Segoe UI", 13),
                bg=DARK_BG_ALT,
                fg=DARK_FG,
                activebackground=PRIMARY,
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                padx=20,
                pady=12,
                width=22,
                command=command
            )
            b.pack(pady=8)
            return b

        self.btn_add = make_button("➕ Ajouter un visage",
                                   lambda: controller.show_frame("AddFaceFrame"))
        self.btn_train = make_button("🧠 Entraîner le modèle",
                                     self.on_train_clicked)
        self.btn_recognize = make_button("👁️ Reconnaissance faciale",
                                         lambda: controller.show_frame("RecognitionFrame"))

        self.status_label = tk.Label(
            self,
            text="Prêt.",
            font=("Segoe UI", 11),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        self.status_label.pack(pady=20)

    def set_status(self, text):
        self.status_label.config(text=text)
        self.status_label.update_idletasks()

    def on_train_clicked(self):
        self.set_status("Entraînement en cours...")
        self.btn_train.config(state="disabled")
        self.controller.update_idletasks()

        ok = train_model(status_callback=self.set_status)

        if not ok:
            messagebox.showwarning("Entraînement",
                                   "L'entraînement n'a pas pu être effectué.\n"
                                   "Vérifie que tu as bien des images dans dataset/.")
        else:
            messagebox.showinfo("Entraînement", "Modèle entraîné avec succès ✅")

        self.btn_train.config(state="normal")


class AddFaceFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller

        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.capturing = False
        self.current_person = None
        self.image_count = 0

        # Top bar
        top = tk.Frame(self, bg=DARK_BG)
        top.pack(fill="x", pady=(10, 10), padx=10)

        back_btn = tk.Button(
            top,
            text="← Retour",
            font=("Segoe UI", 11),
            bg=DARK_BG_ALT,
            fg=DARK_FG,
            relief="flat",
            command=self.go_back
        )
        back_btn.pack(side="left")

        title = tk.Label(
            top,
            text="Ajouter / gérer les visages",
            font=("Segoe UI", 18, "bold"),
            bg=DARK_BG,
            fg=PRIMARY
        )
        title.pack(side="left", padx=15)

        # Main content
        content = tk.Frame(self, bg=DARK_BG)
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Left : liste des personnes
        left = tk.Frame(content, bg=DARK_BG)
        left.pack(side="left", fill="y", padx=(0, 20))

        lbl_existing = tk.Label(
            left, text="Visages existants",
            font=("Segoe UI", 12, "bold"),
            bg=DARK_BG,
            fg=DARK_FG
        )
        lbl_existing.pack(anchor="w", pady=(0, 5))

        self.listbox = tk.Listbox(
            left,
            bg=DARK_BG_ALT,
            fg=DARK_FG,
            selectbackground=PRIMARY,
            relief="flat",
            width=25,
            height=20
        )
        self.listbox.pack(pady=5, fill="y")

        self.refresh_person_list()

        # Right : form + caméra
        right = tk.Frame(content, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)

        form = tk.Frame(right, bg=DARK_BG)
        form.pack(fill="x", pady=(0, 10))

        name_label = tk.Label(
            form,
            text="Nom de la personne :",
            font=("Segoe UI", 11),
            bg=DARK_BG,
            fg=DARK_FG
        )
        name_label.pack(anchor="w")

        self.name_entry = tk.Entry(
            form,
            font=("Segoe UI", 11),
            bg=DARK_BG_ALT,
            fg=DARK_FG,
            relief="flat",
            insertbackground=DARK_FG
        )
        self.name_entry.pack(fill="x", pady=5)

        buttons_bar = tk.Frame(form, bg=DARK_BG)
        buttons_bar.pack(fill="x", pady=(5, 10))

        self.start_btn = tk.Button(
            buttons_bar,
            text="Démarrer la capture",
            font=("Segoe UI", 11),
            bg=PRIMARY,
            fg="#ffffff",
            relief="flat",
            command=self.start_capture
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.capture_btn = tk.Button(
            buttons_bar,
            text="Capturer",
            font=("Segoe UI", 11),
            bg=DARK_BG_ALT,
            fg=DARK_FG,
            relief="flat",
            state="disabled",
            command=self.capture_image
        )
        self.capture_btn.pack(side="left")

        self.info_label = tk.Label(
            form,
            text="Entre un nom, puis clique sur 'Démarrer la capture'.",
            font=("Segoe UI", 10),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        self.info_label.pack(anchor="w", pady=(5, 0))

        # Zone vidéo
        self.video_label = tk.Label(
            right,
            bg=DARK_BG_ALT,
            bd=4,
            relief="ridge",
            width=640,
            height=480
        )
        self.video_label.pack(pady=10)

        self.counter_label = tk.Label(
            right,
            text="Images capturées : 0",
            font=("Segoe UI", 11),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        self.counter_label.pack(anchor="w", pady=(0, 10))

    def refresh_person_list(self):
        self.listbox.delete(0, tk.END)
        if not os.path.isdir(DATASET_DIR):
            return
        people = [
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
        for p in sorted(people):
            self.listbox.insert(tk.END, p)

    def start_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Nom manquant", "Entre un nom avant de démarrer.")
            return

        self.current_person = name
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        existing = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.image_count = len(existing)
        self.counter_label.config(text=f"Images capturées : {self.image_count}")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Caméra",
                                 "Impossible d'ouvrir la caméra.")
            return

        self.capturing = True
        self.capture_btn.config(state="normal")
        self.info_label.config(
            text="Regarde la caméra. Clique sur 'Capturer' pour enregistrer une image."
        )

        self.update_video()

    def update_video(self):
        if not self.capturing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.info_label.config(text="Erreur : pas d'image depuis la caméra.")
            self.after(50, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self.update_video)

    def capture_image(self):
        if not self.capturing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            self.info_label.config(text="Aucun visage détecté, essaie encore.")
            return

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        person_dir = os.path.join(DATASET_DIR, self.current_person)
        os.makedirs(person_dir, exist_ok=True)
        img_path = os.path.join(person_dir, f"{self.image_count}.jpg")
        cv2.imwrite(img_path, face_roi)
        self.image_count += 1
        self.counter_label.config(text=f"Images capturées : {self.image_count}")
        self.info_label.config(text=f"Image enregistrée : {img_path}")

        self.refresh_person_list()

    def stop_capture(self):
        self.capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")

    def go_back(self):
        self.stop_capture()
        self.controller.show_frame("MainMenu")

    def on_show(self):
        self.info_label.config(
            text="Entre un nom, puis clique sur 'Démarrer la capture'."
        )
        self.counter_label.config(text="Images capturées : 0")
        self.name_entry.delete(0, tk.END)
        self.refresh_person_list()

    def on_hide(self):
        self.stop_capture()


class RecognitionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller

        self.cap = None
        self.running = False
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.recognizer = None
        self.label_map = {}

        self.current_theme = {"mode": "dark"}
        self.anim_state = {"running": False}
        self.BRIGHTNESS_THRESHOLD = 80  # < 80 = scène sombre

        # Top bar
        top = tk.Frame(self, bg=DARK_BG)
        top.pack(fill="x", pady=(10, 10), padx=10)

        back_btn = tk.Button(
            top,
            text="← Retour",
            font=("Segoe UI", 11),
            bg=DARK_BG_ALT,
            fg=DARK_FG,
            relief="flat",
            command=self.go_back
        )
        back_btn.pack(side="left")

        title = tk.Label(
            top,
            text="Reconnaissance faciale",
            font=("Segoe UI", 18, "bold"),
            bg=DARK_BG,
            fg=PRIMARY
        )
        title.pack(side="left", padx=15)

        # Info + vidéo
        info_bar = tk.Frame(self, bg=DARK_BG)
        info_bar.pack(fill="x", pady=(0, 5), padx=20)

        self.info_label = tk.Label(
            info_bar,
            text="Initialisation...",
            font=("Segoe UI", 11),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        self.info_label.pack(side="left")

        self.brightness_label = tk.Label(
            info_bar,
            text="Luminosité : --",
            font=("Segoe UI", 11),
            bg=DARK_BG,
            fg="#aaaaaa"
        )
        self.brightness_label.pack(side="right")

        self.video_label = tk.Label(
            self,
            bg=DARK_BG_ALT,
            bd=4,
            relief="ridge",
            width=640,
            height=480
        )
        self.video_label.pack(pady=10)

    def start_theme_transition(self, to_light: bool):
        if self.anim_state["running"]:
            return
        if to_light and self.current_theme["mode"] == "light":
            return
        if (not to_light) and self.current_theme["mode"] == "dark":
            return

        self.anim_state["running"] = True

        from_bg = LIGHT_BG if self.current_theme["mode"] == "light" else DARK_BG
        to_bg = LIGHT_BG if to_light else DARK_BG

        from_rgb = np.array(hex_to_rgb(from_bg), dtype=float)
        to_rgb = np.array(hex_to_rgb(to_bg), dtype=float)

        steps = 20
        delay = 20

        def step(i):
            t = i / steps
            rgb = from_rgb + (to_rgb - from_rgb) * t
            rgb = tuple(int(x) for x in rgb)
            hex_col = rgb_to_hex(rgb)

            self.controller.configure(bg=hex_col)
            self.configure(bg=hex_col)
            self.video_label.configure(bg=hex_col)

            if i < steps:
                self.after(delay, lambda: step(i + 1))
            else:
                if to_light:
                    self.current_theme["mode"] = "light"
                    final_bg = LIGHT_BG
                    final_fg = LIGHT_FG
                else:
                    self.current_theme["mode"] = "dark"
                    final_bg = DARK_BG
                    final_fg = DARK_FG

                self.controller.configure(bg=final_bg)
                self.configure(bg=final_bg)
                self.video_label.configure(bg=final_bg)
                self.info_label.configure(bg=final_bg, fg=final_fg)
                self.brightness_label.configure(bg=final_bg, fg=final_fg)
                self.anim_state["running"] = False

        step(0)

    def on_show(self):
        # Charger modèle + labels
        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            self.info_label.config(
                text="Modèle manquant. Entraîne d'abord le modèle depuis le menu."
            )
            return

        try:
            self.label_map = load_labels(LABELS_PATH)
            if not self.label_map:
                self.info_label.config(text="Aucun label trouvé.")
                return

            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(MODEL_PATH)
        except Exception as e:
            self.info_label.config(text=f"Erreur modèle : {e}")
            return

        if self.face_cascade.empty():
            self.info_label.config(text="Erreur : classificateur de visage.")
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.info_label.config(text="Impossible d'ouvrir la caméra.")
            return

        self.info_label.config(text="Reconnaissance en cours...")
        self.running = True
        self.update_video()

    def on_hide(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")

    def go_back(self):
        self.controller.show_frame("MainMenu")

    def update_video(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.info_label.config(text="Erreur : pas d'image.")
            self.after(50, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        self.brightness_label.config(text=f"Luminosité : {brightness:.1f}")

        is_dark_scene = brightness < self.BRIGHTNESS_THRESHOLD
        self.start_theme_transition(to_light=is_dark_scene)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        self.info_label.config(text=f"Visages détectés : {len(faces)}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label_id, confidence = self.recognizer.predict(face_roi)

            if confidence < 80:
                name = self.label_map.get(label_id, "Inconnu")
                display_name = f"{name} ({confidence:.0f})"
            else:
                display_name = "Inconnu"

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(frame_rgb, display_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self.update_video)

# ---------- APPLICATION ----------

class EFaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E-Face - Reconnaissance faciale")
        self.configure(bg=DARK_BG)
        self.geometry("950x800")
        self.resizable(False, False)

        self.frames = {}

        for F in (MainMenu, AddFaceFrame, RecognitionFrame):
            name = F.__name__
            frame = F(self, self)
            self.frames[name] = frame
            # chaque frame prend tout l'espace de la fenêtre
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("MainMenu")

    def show_frame(self, name):
        frame = self.frames[name]

        # notifier les autres frames qu'ils se cachent
        for fname, f in self.frames.items():
            if f is frame:
                continue
            if hasattr(f, "on_hide"):
                f.on_hide()

        frame.lift()
        if hasattr(frame, "on_show"):
            frame.on_show()

# ---------- MAIN ----------

if __name__ == "__main__":
    app = EFaceApp()
    app.mainloop()
