import csv
import  pydicom
import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
pth='../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_300663/MR/*/*.dcm'

def load_img_pth(pth):
    dicm=pydicom.dcmread(pth)
    img=dicm.pixel_array.astype(np.float32)
    image=(img-np.min(img))/(np.max(img)-np.min(img))*255
    return image

def extract_features(img):
    mean_val=np.mean(img)
    std_val=np.std(img)
    bright_ratio = np.sum(img > 100) / img.size
    dark_ratio = np.sum(img < 20) / img.size
    return mean_val, std_val, bright_ratio, dark_ratio


# def guess_sequence_type(img):
#     mean_val = np.mean(img)
#     std_val = np.std(img)
#     bright_pixels_ratio = np.sum(img > 200) / img.size
#     dark_pixels_ratio = np.sum(img < 30) / img.size
#
#     print(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Bright Ratio: {bright_pixels_ratio:.2f}, Dark Ratio: {dark_pixels_ratio:.2f}")
#
#     if bright_pixels_ratio > 0.05 and std_val > 40:
#         return "DWI"
#     elif bright_pixels_ratio < 0.01 and dark_pixels_ratio > 0.5:
#         return "ADC"
#     elif mean_val > 80 and std_val > 30:
#         return "T2A"
#     else:
#         return "UNKNOWN"

# Test et


# def loop():
#     for pth in glob(path):
#         img = load_img_pth(pth)
#         seq_type = guess_sequence_type(img)
#         print(f"Tahmini sekans türü: {seq_type}")
# loop()


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
