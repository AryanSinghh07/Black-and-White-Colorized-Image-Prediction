import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Frame, Button, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

# ---------------- Constants & Styles -----------------
BG_COLOR = "#f0f2f5"
PRIMARY_COLOR = "#4CAF50"
ACCENT_COLOR = "#FF8C00"
SUCCESS_COLOR = "#2E8B57"
ERROR_COLOR = "#D32F2F"
TEXT_COLOR = "#333"
BUTTON_HOVER_COLOR = "#388E3C"
FONT_TITLE = ("Helvetica", 24, "bold")
FONT_LABEL = ("Helvetica", 14)
FONT_STATUS = ("Helvetica", 12)
FONT_BUTTON = ("Helvetica", 14, "bold")

# ---------------- Utility Functions -----------------
def show_error(msg):
    messagebox.showerror("Error", msg)

def resource_path(relative_path):
    # For PyInstaller compatibility
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- Image Colorization Logic -----------------
def colorize_image(input_path):
    try:
        prototxt = resource_path("models/colorization_deploy_v2.prototxt")
        model = resource_path("models/colorization_release_v2.caffemodel")
        points = resource_path("models/pts_in_hull.npy")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        pts = np.load(points)
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Could not load image")
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        L_resized = cv2.resize(L, (224, 224))
        L_resized -= 50
        net.setInput(cv2.dnn.blobFromImage(L_resized))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = L[:, :, np.newaxis]
        lab_output = np.concatenate((L, ab), axis=2)
        bgr_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
        bgr_output = np.clip(bgr_output, 0, 1)
        output_image = (bgr_output * 255).astype("uint8")
        return image, output_image
    except Exception as e:
        raise RuntimeError(f"Colorization failed: {e}")

# ---------------- UI Functions -----------------
def set_status(msg, color=TEXT_COLOR):
    status_label.config(text=msg, fg=color)
    status_label.update_idletasks()

def show_spinner():
    spinner_label.place(relx=0.5, rely=0.5, anchor="center")
    spinner_label.lift()
    spinner_label.update_idletasks()

def hide_spinner():
    spinner_label.place_forget()
    spinner_label.update_idletasks()

def save_colorized_image():
    if hasattr(save_colorized_image, 'img'):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg")])
        if file_path:
            Image.fromarray(save_colorized_image.img).save(file_path)
            set_status("Image saved!", SUCCESS_COLOR)
    else:
        set_status("No colorized image to save.", ERROR_COLOR)

def process_image(file_path):
    try:
        set_status("Processing...", ACCENT_COLOR)
        show_spinner()
        original, colorized = colorize_image(file_path)
        # Resize to 400x400
        original_resized = cv2.resize(original, (400, 400))
        colorized_resized = cv2.resize(colorized, (400, 400))
        original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        colorized_rgb = cv2.cvtColor(colorized_resized, cv2.COLOR_BGR2RGB)
        original_image_tk = ImageTk.PhotoImage(Image.fromarray(original_rgb))
        colorized_image_tk = ImageTk.PhotoImage(Image.fromarray(colorized_rgb))
        original_label.config(image=original_image_tk)
        original_label.image = original_image_tk
        colorized_label.config(image=colorized_image_tk)
        colorized_label.image = colorized_image_tk
        save_colorized_image.img = colorized_rgb
        save_button.config(state="normal")
        set_status("Colorization Complete!", SUCCESS_COLOR)
    except Exception as e:
        show_error(str(e))
        set_status("Failed to colorize image.", ERROR_COLOR)
    finally:
        hide_spinner()

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        save_button.config(state="disabled")
        threading.Thread(target=process_image, args=(file_path,), daemon=True).start()

def on_enter(e):
    upload_button.config(bg=BUTTON_HOVER_COLOR)
def on_leave(e):
    upload_button.config(bg=PRIMARY_COLOR)
def on_save_enter(e):
    save_button.config(bg=BUTTON_HOVER_COLOR)
def on_save_leave(e):
    save_button.config(bg=PRIMARY_COLOR)

# ---------------- Main UI Setup -----------------
root = tk.Tk()
root.title("🎨 AI Image Colorizer")
root.configure(bg=BG_COLOR)
root.geometry("900x720")
root.resizable(False, False)

# Logo/Icon (simple colored circle as placeholder)
logo_canvas = tk.Canvas(root, width=80, height=80, bg=BG_COLOR, highlightthickness=0)
logo_canvas.create_oval(10, 10, 70, 70, fill=PRIMARY_COLOR, outline="")
logo_canvas.place(x=30, y=10)

title = Label(root, text="AI Black & White Photo Colorizer Prediction", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR)
title.pack(pady=(30, 20))

frame = Frame(root, bg=BG_COLOR)
frame.pack()

# Image labels
image_titles = Frame(root, bg=BG_COLOR)
image_titles.pack(pady=(0, 5))
Label(image_titles, text="Original", font=FONT_LABEL, bg=BG_COLOR).grid(row=0, column=0, padx=195)
Label(image_titles, text="Colorized", font=FONT_LABEL, bg=BG_COLOR).grid(row=0, column=1, padx=195)

original_label = Label(frame, bd=2, relief="solid", bg="white", width=400, height=400)
original_label.grid(row=0, column=0, padx=15, pady=10)
colorized_label = Label(frame, bd=2, relief="solid", bg="white", width=400, height=400)
colorized_label.grid(row=0, column=1, padx=15, pady=10)

# Spinner (hidden by default)
spinner_img = ImageTk.PhotoImage(Image.new("RGBA", (60, 60), (0,0,0,0)))
spinner_label = Label(root, image=spinner_img, bg=BG_COLOR)
spinner_label.place_forget()

status_label = Label(root, text="Upload a black and white image to colorize", font=FONT_STATUS, bg=BG_COLOR, fg=TEXT_COLOR)
status_label.pack(pady=10)

upload_button = Button(root, text="📤 Upload Image", command=open_image,
                       font=FONT_BUTTON, bg=PRIMARY_COLOR, fg="white", padx=20, pady=10, bd=0, relief="ridge",
                       activebackground=BUTTON_HOVER_COLOR, activeforeground="white", cursor="hand2")
upload_button.pack(pady=15)
upload_button.bind("<Enter>", on_enter)
upload_button.bind("<Leave>", on_leave)

save_button = Button(root, text="💾 Save Colorized Image", command=save_colorized_image,
                     font=FONT_BUTTON, bg=PRIMARY_COLOR, fg="white", padx=20, pady=10, bd=0, relief="ridge",
                     activebackground=BUTTON_HOVER_COLOR, activeforeground="white", cursor="hand2", state="disabled")
save_button.pack(pady=5)
save_button.bind("<Enter>", on_save_enter)
save_button.bind("<Leave>", on_save_leave)

# Placeholder preview
placeholder = ImageTk.PhotoImage(Image.new("RGB", (400, 400), (240, 242, 245)))
original_label.config(image=placeholder)
original_label.image = placeholder
colorized_label.config(image=placeholder)
colorized_label.image = placeholder

root.mainloop()