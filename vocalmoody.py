import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report

# Path dataset
DATASET_PATH = r"C:\Users\sulthon chaidir ali\Downloads\dataset"
print("Dataset ada?", os.path.exists(DATASET_PATH))



# Label emosi → angka (regresi)
emotion_labels = {
    "angry": 0,
    "sad": 1,
    "neutral": 2,
    "happy": 3
}

X = []
y = []

# Ekstraksi MFCC
def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Load data
for emotion, label in emotion_labels.items():
    folder = os.path.join(DATASET_PATH, emotion)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            mfcc_features = extract_mfcc(file_path)
            X.append(mfcc_features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Regresi
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred_continuous = model.predict(X_test)

# Konversi ke label terdekat
y_pred = np.floor(y_pred_continuous + 0.5)
y_pred = np.clip(y_pred, 0, 3).astype(int)


# Evaluasi
print("MSE:", mean_squared_error(y_test, y_pred_continuous))
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=list(emotion_labels.keys()),
    zero_division=0
))

