import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def img_binarize(img_in ):
    shape = img_in.shape
    binimg = np.zeros(shape )
    for i in range(shape[0]):
        for j in range(shape[1]):
           
                if img_in[i][j] >= 128:
                    binimg[i][j] =255
                else:
                    binimg[i][j] = 0
    return binimg
# 3-5-5-5-3
kernel =     [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]
#hit and miss kernel
kernel_j = [[0, -1], [0, 0], [1, 0]]

def dilation(img_in , kernel):
    shape = img_in.shape
    img_dil = np.zeros(img_in.shape , dtype = int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_in[i][j] > 0:
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)>=0 and(i+kernel_i)<=(shape[0]-1)and \
                        (j+kernel_j)>=0 and(j+kernel_j)<=(shape[1]-1):
                            
                        img_dil[i+kernel_i][j+kernel_j]=255
    return img_dil

def erosion(img_in , kernel):
    shape = img_in.shape
    img_ero = np.zeros(img_in.shape , dtype = int)
    for i in range(shape[0]):
        for j in range(shape[1]):
          
                img_ero[i, j] = 255
                ok = 1
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)<0 or(i+kernel_i)>=shape[0]or \
                        (j+kernel_j)<0 or(j+kernel_j)>=shape[1] or img_in[i+kernel_i , j+kernel_j]!=255:   
                        ok =0
                        break
                if ok ==0:
                    img_ero[i, j] = 0
                    
            
    return img_ero
def opening(img_in ,kernel ):
    img_open = erosion(dilation(img_in , kernel), kernel)
    return img_open
    
    
def closing(img_in , kernel):
    img_close = dilation(erosion(img_in , kernel), kernel)
    return img_close

def hit_and_miss(a, j, k):
    b = -a + 255
    return (((erosion(a, j) + erosion(b, k)) // 2) == 255) * 255


img_in = cv2.imread('lena.bmp', 2)
bin_img = img_binarize(img_in)
img_dil = dilation(bin_img , kernel)
cv2.imwrite('img_dil.png' , img_dil)
img_ero = erosion(bin_img, kernel)
cv2.imwrite('img_ero.png' , img_ero)
img_open = opening(bin_img, kernel)
cv2.imwrite('img_open.png' , img_open)
img_close = closing(bin_img, kernel) 
cv2.imwrite('img_close.png' , img_close)   

img_hm = hit_and_miss(bin_img, kernel_j, kernel)
cv2.imwrite('lena_hm_cv.png', img_hm)  

