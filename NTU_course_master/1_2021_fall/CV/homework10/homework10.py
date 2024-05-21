import numpy as np
import cv2 
#################################
img = cv2.imread('lena.bmp', 2)
#################################

def convolving (img ,k):
    rowimg , colimg = img.shape
    rowk ,colk =k.shape
    convolving = 0
    for i in range (rowimg):
        for j in range(colimg):
            if rowimg- i- 1>= 0 and rowimg-i -1 <rowk and\
               colimg- j-1>= 0 and colimg -j-1 <colk:
                    convolving +=(img[i,j] *k[rowimg-i -1,colimg -j-1])
    return convolving
def laplace(img, threshold , c):
    kernel1 = np.array([[0,1, 0],
                       [1,-4,1],
                       [0,1, 0]])
    kernel2 = np.array([[1,1, 1],
                        [1,-8,1],
                        [1,1, 1]])/3
    kernel3 = np.array([[2, -1,  2],
                        [-1,-4, -1],
                        [2, -1, 2]])/3
    kernel4 = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
                        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
                        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
    kernel5 = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
                        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])
    row ,col = img.shape
    rk1 ,rc1 = kernel1.shape
    rk2 ,rc2 = kernel2.shape
    rk3 ,rc3 = kernel3.shape
    rk4 ,rc4 = kernel4.shape
    rk5 ,rc5 = kernel5.shape
    if c ==1:
        imglap = np.zeros((row-rk1+1,col-rc1+1))
        ri ,ci = imglap.shape
        for i in range(ri):
            for j in range(ci):
                tmp = convolving(img[i:i+rk1,j:j+rc1], kernel1)
                if tmp >= threshold:
                    imglap[i,j] = 0
                else:
                    imglap[i,j] = 255
        return imglap
    elif c ==2:
        imglap = np.zeros((row-rk2+1,col-rc2+1))
        ri ,ci = imglap.shape
        for i in range(ri):
            for j in range(ci):
                tmp = convolving(img[i:i+rk2,j:j+rc2], kernel2)
                if tmp > threshold:
                    imglap[i,j] = 0
                
                else:
                    imglap[i,j] = 255
        return imglap
    elif c ==3:
        imglap = np.zeros((row-rk3+1,col-rc3+1))
        ri ,ci = imglap.shape
        for i in range(ri):
            for j in range(ci):
                tmp = convolving(img[i:i+rk3,j:j+rc3], kernel3)
                if tmp > threshold:
                    imglap[i,j] = 0
                
                else:
                    imglap[i,j] = 255
        return imglap
    elif c ==4:
        imglap = np.zeros((row-rk4+1,col-rc4+1))
        ri ,ci = imglap.shape
        for i in range(ri):
            for j in range(ci):
                tmp = convolving(img[i:i+rk4,j:j+rc4], kernel4)
                if tmp > threshold:
                    imglap[i,j] = 0
                
                else:
                    imglap[i,j] = 255
        return imglap
    elif c ==5:
        imglap = np.zeros((row-rk5+1,col-rc5+1))
        ri ,ci = imglap.shape
        for i in range(ri):
            for j in range(ci):
                tmp = convolving(img[i:i+rk5,j:j+rc5], kernel5)
                if tmp >=  threshold:
                    imglap[i,j] = 255
                elif tmp <= -threshold:
                    imglap[i,j] = -255
                else:
                    imglap[i,j] = 0
        return imglap
   
cv2.imwrite('1Lena_laplace_1.bmp',laplace(img, 15, 1))
cv2.imwrite('1Lena_laplace_2.bmp',laplace(img, 15, 2))
cv2.imwrite('2Lena_min_laplace.bmp',laplace(img, 20, 3))
cv2.imwrite('3Lena_laplace_gaussian.bmp',laplace(img, 3000, 4))
cv2.imwrite('4Lena_dif_of_gaussian.bmp',laplace(img, 1, 5))
    