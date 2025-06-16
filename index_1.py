#YORUM VE BILGILENDRME 13.06.2025
#DICOM dosyalarini micro dicom viewer ile okuyabilirsiniz.
#Ornek bir dicom dosyasini alip sayisal bir diziye cevirdim sonra normalize ettim
#Once virgullu sayi formuna sonra da pozitif forma soktum ve 8 bit bir resim olarak kaydettim .
#Bu forma soktuktan sonra pillow_img.show() ile resim olarak goruntuleyebilirsiniz.
#AMA burada cok buyuk detay kaybi oluyor bu da istenilen sonuclari vermeyebilir.
#surec icinde daha cok iyilestirme icin calisacagim.

#YORUM v BILGILENDRME 16.06.2025
#1 line 107
#1.2 fotolori mp4'e cevirerek scan.mp4 adinda bir video olustrudm bu video ile bir zaman akisi olusabilir ve gozlemleyebilirsiniz

#2 resimleri goruntuleme ve kaydetme kodlarini def icine aldim altina yorum olarak koydum
    # yorum satirini kaldirarak kullanabilirsiniz

#line 117
#3 belli bir klasordeki dicom dosyalarini png ye cevirme

import os
import cv2
from glob import glob
import numpy as np
import pydicom
from pydicom import dcmread
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image

kanama_liste=[100236,104541,109022,109992]

fpath=f'../../Iskemi/Iskemi_veri_kumesi/Kanama2/100236.dcm'
dc = dcmread(fpath)
dc_arr = dc.pixel_array

plt_dir = '../../knma2_plt/'
def convert_dcm(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(len(kanama_liste)):
        #Get the file path for a test data file
        newpath=f'../../Iskemi/Iskemi_veri_kumesi/Kanama2/{kanama_liste[i]}.dcm'

        #READ DICOM
        dc=dcmread(newpath)
        dc_newarr=dc.pixel_array
        #showin
        plt.imsave(f'{dir}/{kanama_liste[i]}.jpeg',dc_newarr,cmap='hot')
        plt.imshow(dc_newarr)
        plt.show()
# convert_dcm(plt_dir)

#NORMALIZE DICOM FILE NUMERICALLY
# float_dcom_arr=dc_arr.astype(float)#Dicom array converted to float
# positive_dcom_arr=np.maximum(float_dcom_arr,0)#now positive numbers
# normalized_dcom_arr=positive_dcom_arr/positive_dcom_arr.max()
# normalized_dcom_arr *= 255.0
# uint8_img=np.uint8(normalized_dcom_arr)#now 8 bit image
# pillow_img=Image.fromarray(uint8_img)
# # foo= np.unique(uint8_img) #you can check the numeric values of image and unique values
# # pillow_img.show()#there's big loss compared to original image

#DEF2 for normalizing vision


#######################################################################################
##SECOND VERSION 2nd Normalizing
######################################################################################
jpeg_dir='../../Kanama2_jpeg/'
def dcm_to_jpg(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(len(kanama_liste)):
        newpath=fpath=f'../../Iskemi/Iskemi_veri_kumesi/Kanama2/{kanama_liste[i]}.dcm'
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
        pillow_img2.save(f'{dir}/{kanama_liste[i]}.jpeg', 'JPEG',quality=100)
        pillow_img2.show()
# dcm_to_jpg(jpeg_dir) #images pop up

def normalize_visualization_dicom(dcm,max_v=None,min_v=None,show=False):
    dicom_file=dcmread(dcm)
    dcm_arr=dicom_file.pixel_array.astype(float)
    if max_v: hounsefiel_max = max_v
    else: hounsefiel_max=np.max(dcm_arr)

    if min_v: hounsefiel_min=min_v
    else: hounsefiel_min=np.min(dcm_arr)

    hounsefiel_range=hounsefiel_max-hounsefiel_min

    dcm_arr[dcm_arr < hounsefiel_min] = hounsefiel_min
    dcm_arr[dcm_arr>hounsefiel_max]=hounsefiel_max
    normalized_dcom_arr2=((dcm_arr - hounsefiel_min)/hounsefiel_range)*255.0
    uint8_img_2=np.uint8(normalized_dcom_arr2)
    if show:
        pillow_image=Image.fromarray(uint8_img_2)
        # pillow_image.show()
    return uint8_img_2


def dicom_to_png(dcm_file,pth,max_v,min_v):
    path_name=os.path.basename(dcm_file)[:-4]
    array=normalize_visualization_dicom(dcm_file,max_v,min_v,False)
    img=Image.fromarray(array)
    img.save(f'{pth}{path_name}.png')


dicom_f_pth=glob('../../Iskemi/Iskemi_veri_kumesi/Kanama2/*.dcm')
def dicom_to_png_all(f_pth):
    if not os.path.exists('./data_image/'):
        os.makedirs('./data_image/')
    for dcm_file in f_pth:
        dicom_to_png(dcm_file,'./data_image/',2000,-1000)
# dicom_to_png_all(dicom_f_pth)



#CONVERT DICOM TO MP4
dicom_file_path=glob('../../Iskemi/Iskemi_veri_kumesi/Kanama2/*.dcm')
dicom_file_path.sort()
frame_per_second=3
frame_size=pydicom.dcmread(dicom_file_path[0]).pixel_array.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('scan.mp4', fourcc, frame_per_second, frame_size)

for file in dicom_file_path:
    frame=normalize_visualization_dicom(file,max_v=1600,min_v=-1600)
    cv2_img=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    out.write(cv2_img)
    cv2.imshow('frame',cv2_img)
    cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
