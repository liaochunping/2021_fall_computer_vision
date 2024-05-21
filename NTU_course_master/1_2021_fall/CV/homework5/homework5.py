import cv2
import numpy as np

# 3-5-5-5-3
kernel =     [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]


def dilation(img_in , kernel):
    shape = img_in.shape
    img_dil = np.zeros(img_in.shape , dtype = int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_in[i][j] > 0:
                maxv= 0 #設定成0
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)>=0 and(i+kernel_i)<=(shape[0]-1)and \
                        (j+kernel_j)>=0 and(j+kernel_j)<=(shape[1]-1):
                            if img_in[i+kernel_i,j+kernel_j]>maxv:
                                maxv = (img_in[i+kernel_i,j+kernel_j])
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)>=0 and(i+kernel_i)<=(shape[0]-1)and \
                        (j+kernel_j)>=0 and(j+kernel_j)<=(shape[1]-1):
                            img_dil[i+kernel_i,j+kernel_j]=maxv         
                
    return img_dil

def erosion(img_in , kernel):
    shape = img_in.shape
    img_ero = np.zeros(img_in.shape , dtype = int)
    for i in range(shape[0]):
        for j in range(shape[1]):
          
                img_ero[i, j] = 255
                minv = np.inf #設定成無限大
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)>=0 and(i+kernel_i)<shape[0]and \
                        (j+kernel_j)>=0 and(j+kernel_j)<shape[1]: 
                         if img_in[i+kernel_i,j+kernel_j]<minv:
                            minv =(img_in[i+kernel_i,j+kernel_j])
                for kernel_each in kernel:
                    kernel_i ,kernel_j = kernel_each
                    if (i+kernel_i)>=0 and(i+kernel_i)<shape[0]and \
                        (j+kernel_j)>=0 and(j+kernel_j)<shape[1]:   
                        img_ero[i+kernel_i,j+kernel_j] =minv
    return img_ero
def opening(img_in ,kernel ):
    img_open = erosion(dilation(img_in , kernel), kernel)
    return img_open
    
    
def closing(img_in , kernel):
    img_close = dilation(erosion(img_in , kernel), kernel)
    return img_close


bin_img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
img_dil = dilation(bin_img , kernel)
cv2.imwrite('img_dil.png' , img_dil)
img_ero = erosion(bin_img, kernel)
cv2.imwrite('img_ero.png' , img_ero)
img_open = opening(bin_img, kernel)
cv2.imwrite('img_open.png' , img_open)
img_close = closing(bin_img, kernel) 
cv2.imwrite('img_close.png' , img_close)   
