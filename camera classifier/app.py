import tkinter as tk
from tkinter import simpledialog, messagebox   
import cv2, os
import PIL.Image, PIL.ImageTk
import camera, model

class App:
    def __init__(self, window=tk.Tk(), window_title='Camera Classifier'):
        self.window = window
        self.window_title = window_title
        self.counters = [1, 1]
        self.model = model.Model()
        self.auto_predict = False
        self.camera = camera.Camera()
        self.init_gui()
        self.delay = 15
        self.update()
        self.window.attributes('-topmost', True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=int(self.camera.width), height=int(self.camera.height))
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text='Auto Predict', width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring('Classname One', 'Enter the name for Class 1:', parent=self.window)
        self.classname_two = simpledialog.askstring('Classname Two', 'Enter the name for Class 2:', parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)
        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text='Train Model', width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text='Predict', width=50, command=self.predict, state=tk.DISABLED)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text='Reset Counters', width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text='CLASS', font=('Arial', 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def train_model(self):
        ok = self.model.train(self.counters)
        if ok:
            self.btn_predict.config(state=tk.NORMAL)
            self.class_label.config(text='Trained ✓')
        else:
            messagebox.showwarning("Training", "Training failed or insufficient data. Please capture images for BOTH classes.")

    def auto_predict_toggle(self):
        if not self.model.is_trained:
            messagebox.showinfo("Auto Predict", "Please train the model first.")
            self.auto_predict = False
            return
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_number):
        ret, frame = self.camera.get_frame()
        if not ret:
            return
        if not os.path.exists('1'):
            os.mkdir('1')
        if not os.path.exists('2'):
            os.mkdir('2')

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (150, 150), interpolation=cv2.INTER_AREA)

        idx = self.counters[class_number - 1]
        out_path = f'{class_number}/frame{idx}.jpg'
        cv2.imwrite(out_path, gray)

        self.counters[class_number - 1] += 1
        if self.model.is_trained:
            self.model.is_trained = False
            self.btn_predict.config(state=tk.DISABLED)
            self.class_label.config(text='New data added → retrain')

    def update(self):
        if self.auto_predict and self.model.is_trained:
            self.predict()

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        pred = self.model.predict(self.camera.get_frame())
        if pred is None:
            messagebox.showinfo("Predict", "Please train the model first.")
            return
        if pred == 1:
            self.class_label.config(text=self.classname_one)
        elif pred == 2:
            self.class_label.config(text=self.classname_two)

    def reset(self):
        for directory in ['1', '2']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        self.counters = [1, 1]
        self.model = model.Model()
        self.btn_predict.config(state=tk.DISABLED)
        self.class_label.config(text='CLASS')
