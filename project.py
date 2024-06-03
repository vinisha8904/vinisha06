import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        # Create a label to display the video feed or selected image
        self.label = tk.Label(self.root)
        self.label.pack()

        # Create a label to display the number of faces detected
        self.face_count_label = tk.Label(self.root, text="Faces Detected: 0")
        self.face_count_label.pack()

        # Create a dropdown menu for selecting filters
        self.filter_var = tk.StringVar()
        self.filter_var.set("None")
        self.filter_menu = tk.OptionMenu(self.root, self.filter_var, "None", "Grayscale", "Blur", command=self.apply_filter)
        self.filter_menu.pack()

        # Create a button to select an image
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack()

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)
        self.selected_image = None
        self.faces = None  # Initialize faces variable

        # Start the video feed
        self.show_frame()

    def show_frame(self):
        if self.selected_image is None:
            _, frame = self.cap.read()

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            self.faces = faces  # Store faces in instance variable

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Update face count label
            self.face_count_label.config(text=f"Faces Detected: {len(faces)}")

            # Convert the frame back to PIL format
            img = Image.fromarray(rgb_frame)
        else:
            # Detect faces in the selected image
            img = self.selected_image.copy()
            gray = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            self.faces = faces  # Store faces in instance variable

            # Draw rectangles around the detected faces
            draw = ImageDraw.Draw(img)
            for (x, y, w, h) in faces:
                draw.rectangle([x, y, x+w, y+h], outline="red", width=2)

            # Update face count label
            self.face_count_label.config(text=f"Faces Detected: {len(faces)}")

        # Convert PIL image to Tkinter format and display it
        img_tk = ImageTk.PhotoImage(image=img)
        self.label.img = img_tk
        self.label.config(image=img_tk)

        # Repeat the process after 10 milliseconds
        self.root.after(10, self.show_frame)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_image = Image.open(file_path).convert("RGB")

    def apply_filter(self, choice):
        if self.selected_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        # Apply filters to the selected image
        if choice == "Grayscale":
            img = self.selected_image.convert("L")
        elif choice == "Blur":
            img = self.selected_image.filter(ImageFilter.BLUR)
        else:
            # No filter selected or 'None' option chosen
            img = self.selected_image.copy()

        # Convert PIL image to Tkinter format and display it
        img_tk = ImageTk.PhotoImage(image=img)
        self.label.img = img_tk
        self.label.config(image=img_tk)

        # Apply filters to the detected faces
        if choice != "None" and self.faces is not None:
            for (x, y, w, h) in self.faces:
                face_img = img.crop((x, y, x+w, y+h))
                if choice == "Grayscale":
                    face_img = face_img.convert("L")
                elif choice == "Blur":
                    face_img = face_img.filter(ImageFilter.BLUR)
                img.paste(face_img, (x, y, x+w, y+h))

        # Convert PIL image to Tkinter format and display it with applied filters
        img_tk = ImageTk.PhotoImage(image=img)
        self.label.img = img_tk
        self.label.config(image=img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
