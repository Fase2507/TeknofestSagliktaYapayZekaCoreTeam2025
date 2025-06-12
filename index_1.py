import numpy as np
from mpmath.libmp import normalize
from pydicom import dcmread
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt
from torch.distributions.constraints import positive
from PIL import Image

#Get the file path for a test data file
fpath='./Iskemi/Iskemi_veri_kumesi/Diğer (Normal ve Kronik İskemi Bulgu Kesitleri)/100005.dcm'

#READ DICOM
dc=dcmread(fpath)
rows=dc.Rows
columns=dc.Columns
dc_arr=dc.pixel_array
# plt.imshow(dc_arr)
# plt.show()


#NORMALIZE DICOM FILE NUMERICALLY
float_dcom_arr=dc_arr.astype(float)#Dicom array converted to float
positive_dcom_arr=np.maximum(float_dcom_arr,0)#now positive numbers
normalized_dcom_arr=positive_dcom_arr/positive_dcom_arr.max()
normalized_dcom_arr *= 255.0#not 512 anymore
uint8_img=np.uint8(normalized_dcom_arr)#now 8 bit image
pillow_img=Image.fromarray(uint8_img)
# foo= np.unique(uint8_img)
pillow_img.show()#there's big loss compared to original image
print(pillow_img)

