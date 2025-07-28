import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import pandas as pd
from flower_predictor import FlowerPredictor
import threading

class FlowerPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flower Predictor GUI")
        self.root.geometry("800x600")
        
        self.predictor = FlowerPredictor(model_type='random_forest')
        self.data_dir = 'Flower_dataset'
        self.feedback_file = 'user_feedback.csv'
        
        # Load initial training data and train model
        X, y = self.predictor.load_training_data(self.data_dir)
        if len(X) > 0:
            self.predictor.train_model(X, y)
        else:
            messagebox.showinfo("Info", "No training data found. Using rule-based classification.")
        
        self.selected_image_path = None
        
        self.create_widgets()
        self.load_feedback_data()
    
    def create_widgets(self):
        # Frame for image and prediction
        self.frame_left = tk.Frame(self.root, width=400, height=600)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Frame for controls and feedback
        self.frame_right = tk.Frame(self.root, width=400, height=600)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display label
        self.image_label = tk.Label(self.frame_left, text="No Image Selected", width=50, height=25, bg="gray")
        self.image_label.pack(padx=10, pady=10)
        
        # Button to select image
        self.btn_select_image = tk.Button(self.frame_left, text="Select Image", command=self.select_image)
        self.btn_select_image.pack(pady=5)
        
        # Prediction result labels
        self.label_species = tk.Label(self.frame_right, text="Predicted Species: N/A", font=("Arial", 14))
        self.label_species.pack(pady=10)
        
        self.label_confidence = tk.Label(self.frame_right, text="Confidence: N/A", font=("Arial", 14))
        self.label_confidence.pack(pady=10)
        
        self.label_measurements = tk.Label(self.frame_right, text="Measurements:\nSepal Length: N/A\nSepal Width: N/A\nPetal Length: N/A\nPetal Width: N/A", font=("Arial", 12), justify=tk.LEFT)
        self.label_measurements.pack(pady=10)
        
        # Feedback dropdown label
        self.label_feedback = tk.Label(self.frame_right, text="If prediction is wrong, select correct species:", font=("Arial", 12))
        self.label_feedback.pack(pady=10)
        
        # Dropdown for species feedback
        self.species_var = tk.StringVar()
        self.dropdown_species = ttk.Combobox(self.frame_right, textvariable=self.species_var, state="readonly")
        self.dropdown_species['values'] = sorted(self.predictor.default_measurements.keys())
        self.dropdown_species.pack(pady=5)
        
        # Button to submit feedback
        self.btn_submit_feedback = tk.Button(self.frame_right, text="Submit Feedback and Retrain", command=self.submit_feedback)
        self.btn_submit_feedback.pack(pady=10)
        
        # Button to reset model
        self.btn_reset_model = tk.Button(self.frame_right, text="Reset Model to Original", command=self.reset_model)
        self.btn_reset_model.pack(pady=10)
        
        # Status message label
        self.label_status = tk.Label(self.frame_right, text="", font=("Arial", 12), fg="blue")
        self.label_status.pack(pady=10)
        
        # Button to visualize morphology diagram
        self.btn_visualize = tk.Button(self.frame_right, text="Visualize Morphology Diagram", command=self.visualize_morphology)
        self.btn_visualize.pack(pady=10)
    
    def select_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png")]
        filepath = filedialog.askopenfilename(title="Select Flower Image", filetypes=filetypes)
        if filepath:
            self.selected_image_path = filepath
            self.display_image(filepath)
            self.predict_flower(filepath)
    
    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((380, 380))
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.img_tk, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def predict_flower(self, image_path):
        def task():
            self.label_status.config(text="Predicting...")
            species, confidence, measurements, features = self.predictor.predict_flower(image_path)
            self.current_features = features
            self.label_species.config(text=f"Predicted Species: {species.replace('_', ' ').title()}")
            self.label_confidence.config(text=f"Confidence: {confidence:.2%}")
            meas_text = f"Measurements:\nSepal Length: {measurements.get('sepal_length', 'N/A')}\nSepal Width: {measurements.get('sepal_width', 'N/A')}\nPetal Length: {measurements.get('petal_length', 'N/A')}\nPetal Width: {measurements.get('petal_width', 'N/A')}"
            self.label_measurements.config(text=meas_text)
            self.label_status.config(text="Prediction complete.")
        threading.Thread(target=task).start()
    
    def submit_feedback(self):
        correct_species = self.species_var.get()
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        if not correct_species:
            messagebox.showwarning("Warning", "Please select the correct species from the dropdown.")
            return
        
        # Extract features for the selected image
        features = self.predictor.extract_image_features(self.selected_image_path)
        if features is None:
            messagebox.showerror("Error", "Failed to extract features from the image.")
            return
        
        # Append new training data
        self.predictor.training_data_X.append(features)
        self.predictor.training_data_y.append(correct_species)
        
        # Retrain model in a separate thread
        def retrain_task():
            self.label_status.config(text="Retraining model with feedback...")
            try:
                X = self.predictor.training_data_X
                y = self.predictor.training_data_y
                self.predictor.train_model(X, y)
                self.save_feedback(self.selected_image_path, correct_species)
                self.label_status.config(text="Model retrained successfully with feedback.")
            except Exception as e:
                self.label_status.config(text=f"Retraining failed: {e}")
        
        threading.Thread(target=retrain_task).start()
    
    def save_feedback(self, image_path, species):
        # Save feedback to CSV for persistence
        feedback_data = []
        if os.path.exists(self.feedback_file):
            try:
                feedback_data = pd.read_csv(self.feedback_file).to_dict('records')
            except:
                feedback_data = []
        feedback_data.append({'image_path': image_path, 'species': species})
        df = pd.DataFrame(feedback_data)
        df.to_csv(self.feedback_file, index=False)
    
    def load_feedback_data(self):
        # Load feedback data and retrain model if any
        if os.path.exists(self.feedback_file):
            try:
                df = pd.read_csv(self.feedback_file)
                for _, row in df.iterrows():
                    features = self.predictor.extract_image_features(row['image_path'])
                    if features is not None:
                        self.predictor.training_data_X.append(features)
                        self.predictor.training_data_y.append(row['species'])
                if self.predictor.training_data_X and self.predictor.training_data_y:
                    self.predictor.train_model(self.predictor.training_data_X, self.predictor.training_data_y)
            except Exception as e:
                print(f"Failed to load feedback data: {e}")
    
    def reset_model(self):
        # Reload original training data and retrain model
        def reset_task():
            self.label_status.config(text="Resetting model to original training data...")
            try:
                X, y = self.predictor.load_training_data(self.data_dir)
                self.predictor.train_model(X, y)
                self.predictor.training_data_X = list(X)
                self.predictor.training_data_y = list(y)
                self.label_status.config(text="Model reset to original training data.")
            except Exception as e:
                self.label_status.config(text=f"Reset failed: {e}")
        threading.Thread(target=reset_task).start()
    
    def visualize_morphology(self):
        # Visualize morphology diagram for current prediction
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        try:
            species = self.label_species.cget("text").replace("Predicted Species: ", "").lower().replace(" ", "_")
            measurements_text = self.label_measurements.cget("text")
            # Parse measurements from label text
            measurements = {}
            for line in measurements_text.splitlines()[1:]:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(" ", "_")
                    try:
                        value = float(parts[1].strip())
                    except:
                        value = 3.0
                    measurements[key] = value
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5,5))
            self.predictor.draw_morphology_diagram(ax, measurements)
            plt.title(f"Morphology Diagram: {species.replace('_', ' ').title()}")
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize morphology: {e}")

def main():
    root = tk.Tk()
    app = FlowerPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
