#YORUM VE BILGILENDRME 13.06.2025
#DICOM dosyalarini micro dicom viewer ile okuyabilirsiniz.
#Ornek bir dicom dosyasini alip sayisal bir diziye cevirdim sonra normalize ettim
#Once virgullu sayi formuna sonra da pozitif forma soktum ve 8 bit bir resim olarak kaydettim .
#Bu forma soktuktan sonra pillow_img.show() ile resim olarak goruntuleyebilirsiniz.
#AMA burada cok buyuk detay kaybi oluyor bu da istenilen sonuclari vermeyebilir.
#surec icinde daha cok iyilestirme icin calisacagim.
import os
import numpy as np
from mpmath.libmp import normalize
from pydicom import dcmread
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt
from torch.distributions.constraints import positive
from PIL import Image
kanama_liste=[100236,104541,109022,109992]

fpath=f'./Iskemi/Iskemi_veri_kumesi/Kanama2/100236.dcm'
dc = dcmread(fpath)
dc_arr = dc.pixel_array

plt_dir='./Iskemi/Iskemi_veri_kumesi/Kanama2_plt/'
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
for i in range(len(kanama_liste)):
    #Get the file path for a test data file
    newpath=f'./Iskemi/Iskemi_veri_kumesi/Kanama2/{kanama_liste[i]}.dcm'

    #READ DICOM
    dc=dcmread(newpath)
    dc_newarr=dc.pixel_array
    #showin
    plt.imsave(f'{plt_dir}/{kanama_liste[i]}.jpeg',dc_newarr,cmap='hot')
    plt.imshow(dc_newarr)
    plt.show()


#NORMALIZE DICOM FILE NUMERICALLY
float_dcom_arr=dc_arr.astype(float)#Dicom array converted to float
positive_dcom_arr=np.maximum(float_dcom_arr,0)#now positive numbers
normalized_dcom_arr=positive_dcom_arr/positive_dcom_arr.max()
normalized_dcom_arr *= 255.0
uint8_img=np.uint8(normalized_dcom_arr)#now 8 bit image
pillow_img=Image.fromarray(uint8_img)
# foo= np.unique(uint8_img) #you can check the numeric values of image and unique values
# pillow_img.show()#there's big loss compared to original image



#######################################################################################
##SECOND VERSION 2nd Normalizing
######################################################################################
jpeg_dir='./Iskemi/Iskemi_veri_kumesi/Kanama2_jpeg/'
if not os.path.exists(jpeg_dir):
    os.makedirs(jpeg_dir)
for i in range(len(kanama_liste)):
    newpath=fpath=f'./Iskemi/Iskemi_veri_kumesi/Kanama2/{kanama_liste[i]}.dcm'
    dc_arr=dcmread(newpath).pixel_array
    dc_field_min=0 #np.min(dc_arr)
    dc_field_max=2000#np.max(dc_arr)
    hounsefield_range=dc_field_max-dc_field_min

    dc_arr[dc_arr<dc_field_min]=dc_field_min
    dc_arr[dc_arr>dc_field_max]=dc_field_max
    normalized_dcom_arr2=(dc_arr - dc_field_min)/hounsefield_range
    normalized_dcom_arr2*=255.0
    uint8_img_2=np.uint8(normalized_dcom_arr2)
    #show output
    pillow_img2=Image.fromarray(uint8_img_2)
    pillow_img2.save(f'{jpeg_dir}/{kanama_liste[i]}.jpeg', 'JPEG',quality=100)
    pillow_img2.show()