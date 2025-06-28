import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import  os

adc_path='../../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_301075/CT/Seri2/50240502.1.33.dcm'
dwi_path='../../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi\Vaka_301367\CT\Seri4/50581772.3.15.dcm'
t2_path='../../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi\Vaka_301075\MR\Seri301/50583216.2.42.dcm'


def load_dcm(path):
    ds=pydicom.dcmread(path)
    img=ds.pixel_array
    return img

t2_img = load_dcm(t2_path)
dwi_img = load_dcm(dwi_path)
adc_img = load_dcm(adc_path)

fig,axs=plt.subplots(1,3,figsize=(15,5))

axs[0].imshow(adc_img,cmap='gray')
axs[0].set_title("ADC")
axs[1].imshow(dwi_img,cmap='gray')
axs[1].set_title("DWI")
axs[2].imshow(t2_img,cmap='gray')
axs[2].set_title("T2A")

for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()

