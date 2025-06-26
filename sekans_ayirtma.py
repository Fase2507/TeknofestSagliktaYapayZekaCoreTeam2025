import csv
import  pydicom
import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


pth='../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_30*/*/*/*.dcm'

def load_img_pth(pth):
    dicm=pydicom.dcmread(pth)
    img=dicm.pixel_array.astype(np.float32)
    image=(img-np.min(img))/(np.max(img)-np.min(img))*255
    return image

def extract_features(img):
    mean_val=np.mean(img)
    std_val=np.std(img)
    bright_ratio = np.sum(img > 120) / img.size
    dark_ratio = np.sum(img < 30) / img.size
    return mean_val, std_val, bright_ratio, dark_ratio





def classify(mean, std, bright_ratio, dark_ratio):
    # Bu kurallar basit bir örnektir. Gerçek veriyle ayarlanabilir.
    if bright_ratio < 0.01 and dark_ratio > 0.4:
        return 'ADC'
    elif bright_ratio > 0.05 and std > 45:
        return 'DWI'
    elif mean > 90 and std > 30:
        return 'T2A'
    else:
        return 'UNKNOWN'



with open('dicom_analysis.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'mean', 'std', 'bright_ratio', 'dark_ratio', 'predicted_sequence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for path in glob(pth):
        try:
            img = load_img_pth(path)
            mean, std, bright, dark = extract_features(img)
            sequence = classify(mean, std, bright, dark)

            writer.writerow({
                'filename': os.path.basename(path),
                'mean': round(mean, 2),
                'std': round(std, 2),
                'bright_ratio': round(bright, 4),
                'dark_ratio': round(dark, 4),
                'predicted_sequence': sequence
            })
        except Exception as e:
            print(f"HATA ({path}): {e}")

df=pd.read_csv('dicom_analysis.csv')
df=df[df['predicted_sequence']!='UNKNOWN']
X = df[['mean', 'std', 'bright_ratio', 'dark_ratio']]
y=df['predicted_sequence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'sequence_classifier.pkl')


print("Eğitim tamamlandı. Doğruluk:", model.score(X_test, y_test))
