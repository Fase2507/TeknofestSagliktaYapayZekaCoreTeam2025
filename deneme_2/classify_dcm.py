import joblib
import pydicom
import cv2
import numpy as np
import os

# Load trained models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature extraction
def extract_features(img):
    img = img.astype(np.float32)
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    mean = np.mean(img)
    std = np.std(img)
    edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
    edge_density = (edges > 0).mean()
    csf_ratio = (img > 0.80).mean()
    dark_ratio = (img < 0.10).mean()

    return [mean, std, edge_density, csf_ratio, dark_ratio]

# Classify a DICOM file
def classify_dcm(pth):
    dcm = pydicom.dcmread(pth)
    img = dcm.pixel_array
    if img.ndim > 2:
        img = img[0]
    features = np.array([extract_features(img)])  # must be 2D array
    scaled = scaler.transform(features)
    label = kmeans.predict(scaled)[0]

    cluster_to_label = {0: "T2A", 1: "DWI", 2: "ADC"}  # Adjust after inspecting clusters
    return cluster_to_label.get(label, "Unknown")


pth = "../../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_300663/MR/Seri3/50467107.2.14.dcm"
print(f"{pth} → {classify_dcm(pth)}")
