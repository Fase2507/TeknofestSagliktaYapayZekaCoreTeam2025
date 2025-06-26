import csv
import os
import joblib
import shutil
import pydicom
import numpy as np
from glob import glob
import pandas as pd
from pyexpat import features

# Yükle model
model = joblib.load('sequence_classifier.pkl')



# Özellik çıkarımı
def normalize_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img


def extract_features(img):
    mean_val = np.mean(img)
    std_val = np.std(img)
    bright_ratio = np.sum(img > 200) / img.size
    dark_ratio = np.sum(img < 30) / img.size
    return [mean_val, std_val, bright_ratio, dark_ratio]



# Yeni DICOM dosyalarını tara
input_folder = '../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_310*/*/*'
output_base = './classified'
os.makedirs(output_base, exist_ok=True)
dicom_files = glob(os.path.join(input_folder, '*.dcm'))
with open('sekans.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'mean', 'std', 'bright_ratio', 'dark_ratio', 'predicted_sequence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for path in dicom_files:
        try:
            img = normalize_dicom(path)
            features = extract_features(img)
            prediction = model.predict([features])[0]

            # Hedef klasörü oluştur
            target_folder = os.path.join(output_base, prediction)
            os.makedirs(target_folder, exist_ok=True)

            # Kopyala
            shutil.copy(path, os.path.join(target_folder, os.path.basename(path)))
            print(f"{os.path.basename(path)} → {prediction}")
        except Exception as e:
            print(f"Hata ({path}): {e}")

        writer.writerow({
            'filename': os.path.basename(path),
            'mean': round(features[0], 2),
            'std': round(features[1], 2),
            'bright_ratio': round(features[2], 4),
            'dark_ratio': round(features[3], 4),
            'predicted_sequence': prediction
        })