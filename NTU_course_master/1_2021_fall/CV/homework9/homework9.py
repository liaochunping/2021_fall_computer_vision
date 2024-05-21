import numpy as np
import cv2  ,math


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

def Roberts(img ,threshold):
    img = np.asarray(img)
    row ,col = img.shape
    kernel1 = np.array([[-1 ,0],
                         [0,1]])
    kernel2 = np.array([[0 ,-1],
                         [1,0]])
    new1 = np.zeros(img.shape,dtype=int)
    new2 = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            new1[i,j] = convolving(img[i:i+2,j:j+2], kernel1)
            new2[i,j] = convolving(img[i:i+2,j:j+2], kernel2)
    C = np.sqrt(new1**2+new2**2)
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 
def Prewitt(img,threshold):
    img = np.asarray(img)
    row ,col = img.shape
    kernel1 = np.array([[ -1,-1,-1],
                        [  0, 0, 0],
                        [  1, 1, 1]])
    kernel2 = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    new1 = np.zeros(img.shape,dtype=int)
    new2 = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            new1[i,j] = convolving(img[i:i+3,j:j+3], kernel1)
            new2[i,j] = convolving(img[i:i+3,j:j+3], kernel2)
    C = np.sqrt(new1**2+new2**2)
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 
def Sobel(img,threshold):
    img = np.asarray(img)
    row ,col = img.shape
    kernel1 = np.array([[ -1,-2,-1],
                        [  0, 0, 0],
                        [  1, 2, 1]])
    kernel2 = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    new1 = np.zeros(img.shape,dtype=int)
    new2 = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            new1[i,j] = convolving(img[i:i+3,j:j+3], kernel1)
            new2[i,j] = convolving(img[i:i+3,j:j+3], kernel2)
    C = np.sqrt(new1**2+new2**2)
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 
def FreiAndChen(img,threshold):
    img = np.asarray(img)
    row ,col = img.shape
    kernel1 = np.array([[ -1,-np.sqrt(2),-1],
                        [  0, 0, 0],
                        [  1, np.sqrt(2), 1]])
    kernel2 = np.array([[-1,0,1],
                        [-np.sqrt(2),0,np.sqrt(2)],
                        [-1,0,1]])
    new1 = np.zeros(img.shape,dtype=int)
    new2 = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            new1[i,j] = convolving(img[i:i+3,j:j+3], kernel1)
            new2[i,j] = convolving(img[i:i+3,j:j+3], kernel2)
    C = np.sqrt(new1**2+new2**2)
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 



def Kirsch(img, threshold):
    img = np.array(img)
    row ,col = img.shape
    k0 = np.array([[-3, -3, 5],
                    [-3, 0, 5],
                    [-3, -3, 5]])
    k1 = np.array([[-3, 5, 5],
                    [-3, 0, 5],
                    [-3, -3, -3]])
    k2 = np.array([ [5, 5, 5],
                    [-3, 0, -3],
                    [-3, -3, -3]])
    k3 = np.array([[5, 5, -3],
                    [5, 0, -3],
                    [-3, -3, -3]])
    k4 = np.array([ [5, -3, -3],
                    [5, 0, -3],
                    [5, -3, -3]])
    k5 = np.array([[-3, -3, -3],
                    [5, 0, -3],
                    [5, 5, -3]])
    k6 = np.array([[-3, -3, -3],
                    [-3, 0, -3],
                    [5, 5, 5]])
    k7 = np.array([[-3, -3, -3],
                    [-3, 0, 5],
                    [-3, 5, 5]])
    C = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            r0 =convolving(img[i:i+3 ,j:j+3] ,k0)
            r1 =convolving(img[i:i+3 ,j:j+3] ,k1)
            r2 =convolving(img[i:i+3 ,j:j+3] ,k2)
            r3 =convolving(img[i:i+3 ,j:j+3] ,k3)
            r4 =convolving(img[i:i+3 ,j:j+3] ,k4)
            r5 =convolving(img[i:i+3 ,j:j+3] ,k5)
            r6 =convolving(img[i:i+3 ,j:j+3] ,k6)
            r7 =convolving(img[i:i+3 ,j:j+3] ,k7)
            C[i,j] = np.max([r0,r1,r2,r3,r4,r5,r6,r7])
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 
def Robinson(img,threshold): 
    img = np.array(img)
    row ,col = img.shape
    k0 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    k1 = np.array([[0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]])
    k2 = np.array([ [1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    k3 = np.array([[2, 1, 0],
                    [1, 0, -1],
                    [0, -1, -2]])
    k4 = np.array([ [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
    k5 = np.array([[0, -1, -2],
                    [1, 0, -1],
                    [2, 1, 0]])
    k6 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
    k7 = np.array([[-2, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 2]])
    C = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            r0 =convolving(img[i:i+3 ,j:j+3] ,k0)
            r1 =convolving(img[i:i+3 ,j:j+3] ,k1)
            r2 =convolving(img[i:i+3 ,j:j+3] ,k2)
            r3 =convolving(img[i:i+3 ,j:j+3] ,k3)
            r4 =convolving(img[i:i+3 ,j:j+3] ,k4)
            r5 =convolving(img[i:i+3 ,j:j+3] ,k5)
            r6 =convolving(img[i:i+3 ,j:j+3] ,k6)
            r7 =convolving(img[i:i+3 ,j:j+3] ,k7)
            C[i,j] = np.max([r0,r1,r2,r3,r4,r5,r6,r7])
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 



def Nevatia_Babu(img,threshold):
    img = np.array(img)
    row ,col = img.shape
    k0 = np.array([[100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100],
                    [0, 0, 0, 0, 0],
                    [-100, -100, -100, -100, -100],
                    [-100, -100, -100, -100, -100]])
    k1 = np.array([[100, 100, 100, 100, 100],
                    [100, 100, 100, 78, -32],
                    [100, 92, 0, -92, -100],
                    [32, -78, -100, -100, -100],
                    [-100, -100, -100, -100, -100] ])
    k2 = np.array([ [100, 100, 100, 32, -100],
                    [100, 100, 92, -78, -100],
                    [100, 100, 0, -100, -100],
                    [100, 78, -92, -100, -100],
                    [100, -32, -100, -100, -100] ])
    k3 = np.array([ [-100, -100, 0, 100, 100],
                    [-100, -100, 0, 100, 100],
                    [-100, -100, 0, 100, 100],
                    [-100, -100, 0, 100, 100],
                    [-100, -100, 0, 100, 100] ])
    k4 = np.array([[-100, 32, 100, 100, 100],
                    [-100, -78, 92, 100, 100],
                    [-100, -100, 0, 100, 100],
                    [-100, -100, -92, 78, 100],
                    [-100, -100, -100, -32, 100] ])
    k5 = np.array([ [100, 100, 100, 100, 100],
                    [-32, 78, 100, 100, 100],
                    [-100, -92, 0, 92, 100],
                    [-100, -100, -100, -78, 32],
                    [-100, -100, -100, -100, -100]])
    C = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            r0 = convolving(img[i:i+5,j:j+5], k0)
            r1 = convolving(img[i:i+5,j:j+5], k1)
            r2 = convolving(img[i:i+5,j:j+5], k2)
            r3 = convolving(img[i:i+5,j:j+5], k3)
            r4 = convolving(img[i:i+5,j:j+5], k4)
            r5 = convolving(img[i:i+5,j:j+5], k5)
            C[i,j] = np.max([r0,r1,r2,r3,r4,r5])
    new_img = np.zeros(img.shape,dtype=int)
    for i in range(row):
        for j in range(col):
            if C[i,j] > threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    return new_img 



                        

cv2.imwrite('1_Lena_robert.bmp',Roberts(img, 12))
cv2.imwrite('2_Lena_prewitt.bmp',Prewitt(img, 24))
cv2.imwrite('3_Lena_sobel.bmp', Sobel(img, 38))
cv2.imwrite('4_Lena_freiandchen.bmp', FreiAndChen(img, 30))
cv2.imwrite('5_Lena_Kirsch.bmp',Kirsch(img, 135))
cv2.imwrite('6_Lena_robinson.bmp',Robinson(img,43))
cv2.imwrite('7_Lena_Nevatia_Babu.bmp',Nevatia_Babu(img,12500))
