import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Define the root directory of your dataset
Root = r"P:/archive"

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
def extract_feature(file_name, mfcc, chroma, mel, contrast, tonnetz, zero_crossing_rate):
    # Initialize an empty result array
    result = np.array([])

    # Load the audio file
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Compute MFCC features from the raw audio
    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        result = np.hstack((result, mfccs_processed))

    # Compute Chroma features from the raw audio
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_processed = np.mean(chroma.T, axis=0)
        result = np.hstack((result, chroma_processed))

    # Compute MEL Spectrogram features from the raw audio
    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mel_processed = np.mean(mel.T, axis=0)
        result = np.hstack((result, mel_processed))

    # Compute Spectral Contrast features from the raw audio
    if contrast:
        contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        contrast_processed = np.mean(contrast.T, axis=0)
        result = np.hstack((result, contrast_processed))

    # Compute Tonnetz features from the raw audio
    if tonnetz:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
        tonnetz_processed = np.mean(tonnetz.T, axis=0)
        result = np.hstack((result, tonnetz_processed))

    # Compute Zero Crossing Rate features from the raw audio
    if zero_crossing_rate:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(X)
        zero_crossing_rate_processed = np.mean(zero_crossing_rate.T, axis=0)
        result = np.hstack((result, zero_crossing_rate_processed))
    
    return result

# Load data from the dataset
def load_data(test_size=0.25):
    x, y = [], []
    for file in glob.glob(os.path.join(Root, "Actor_*/*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True, zero_crossing_rate=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load data
x_train, x_test, y_train, y_test = load_data()

# Normalize input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter Tuning
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'hidden_layer_sizes': [(100,), (200,), (300,)],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(estimator=MLPClassifier(solver='adam', max_iter=1000), param_grid=param_grid, cv=3)
grid_search.fit(x_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Cross-Validation
cv_scores = cross_val_score(best_model, np.vstack((x_train_scaled, x_test_scaled)), np.hstack((y_train, y_test)), cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Evaluate on Test Set
y_pred = best_model.predict(x_test_scaled)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
f1 = f1_score(y_test, y_pred, average=None)
print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))
print("Test Set F1 Score:", f1)

# Save the best model
with open('best_model_for_prediction.sav', 'wb') as f:
    pickle.dump(best_model, f)

# Save the scaler
with open('scaler_for_prediction.sav', 'wb') as f:
    pickle.dump(scaler, f)
