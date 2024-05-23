import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
from tkinter import ttk
from pygame import mixer

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'sad', 'angry', 'suprised', 'neutral']

# Function to extract features from audio files
def extract_feature(file_name, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True, zero_crossing_rate=True):
    result = np.array([])
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        result = np.hstack((result, mfccs_processed))
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_processed = np.mean(chroma.T, axis=0)
        result = np.hstack((result, chroma_processed))
    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mel_processed = np.mean(mel.T, axis=0)
        result = np.hstack((result, mel_processed))
    if contrast:
        contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        contrast_processed = np.mean(contrast.T, axis=0)
        result = np.hstack((result, contrast_processed))
    if tonnetz:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
        tonnetz_processed = np.mean(tonnetz.T, axis=0)
        result = np.hstack((result, tonnetz_processed))
    if zero_crossing_rate:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(X)
        zero_crossing_rate_processed = np.mean(zero_crossing_rate.T, axis=0)
        result = np.hstack((result, zero_crossing_rate_processed))
    return result

# Load the model
with open('P:/Py_Audio/MLP/best_model_for_prediction.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the scaler
with open('P:/Py_Audio/MLP/scaler_for_prediction.sav', 'rb') as f:
    scaler = pickle.load(f)

# Predict emotion
def predict_emotion(file_path):
    feature = extract_feature(file_path)
    feature_scaled = scaler.transform(feature.reshape(1, -1))
    prediction = loaded_model.predict(feature_scaled)
    return prediction[0]

# GUI application
class EmotionRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("500x400")
        self.root.configure(bg='#f0f0f0')

        # Set icon if available
        icon_path = "C:/Users/pavam/Downloads/download.png"
        if os.path.exists(icon_path):
            self.root.iconphoto(False, PhotoImage(file=icon_path))

        # Initialize mixer
        mixer.init()

        # Title Label
        self.title_label = ttk.Label(root, text="Emotion Recognition from Audio", font=("Helvetica", 18, "bold"))
        self.title_label.pack(pady=20)

        # Load Button
        self.load_button = ttk.Button(root, text="Load Audio File", command=self.load_file)
        self.load_button.pack(pady=20)

        # Result Label
        self.result_label = ttk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        # Style Configuration
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 14), padding=10)
        self.style.configure('TLabel', background='#f0f0f0')

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            try:
                mixer.music.load(file_path)
                mixer.music.play()
                emotion = predict_emotion(file_path)
                self.result_label.config(text=f"Predicted Emotion: {emotion.capitalize()}", foreground='blue')
            except Exception as e:
                messagebox.showerror("Error", f"Could not process the file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerApp(root)
    root.mainloop()
