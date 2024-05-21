import numpy as np
import cv2  ,math


#################################
img = cv2.imread('lena.bmp', 2)
#################################
kernel =     [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]

def dilation(img_in, kernel):
    row , col = img_in.shape # original image
    img_dil = np.zeros(img_in.shape, dtype = 'int32')
    
    for i in range(row ):
        for j in range(col):
                
                max_value = 0
                for k_each in kernel:
                    ki, kj = k_each
                    if  i + ki >= 0 and i + ki < row  and j + kj >= 0 and j + kj < col: 
                       
                        max_value = max(max_value, img_in[i + ki, j + kj])
                
                img_dil[i, j] = max_value 
                        
    return img_dil
def erosion(img_in, kernel):
    row , col = img_in.shape # original image
    img_ero = np.zeros(img_in.shape, dtype = 'int32')
    
    for i in range(row ):
        for j in range(col):
                
                min_value = np.inf
                for k_each in kernel:
                    ki, kj = k_each
                    if  i + ki >= 0 and i + ki < row                     and j + kj >= 0 and j + kj < col: 
                        
                        min_value = min(min_value, img_in[i + ki, j + kj])
                    
                img_ero[i, j] = min_value
    
    return img_ero

def opening(img_in ,kernel ):
    img_open = erosion(dilation(img_in , kernel), kernel)
    return img_open
   
def closing(img_in , kernel):
    img_close = dilation(erosion(img_in , kernel), kernel)
    return img_close

def getGaussNoise(img_in , mu ,sigma ,amp):
    return img_in + amp * np.random.normal(mu, sigma, img_in.shape)

def getSaltPepperNoise(img_in ,threshold):
    distribution_map = np.random.uniform( 0,1, img_in.shape)
    temp = np.copy(img_in)  
    row, col = img_in.shape
    for i in range(row):
        for j in range(col):
            if distribution_map[i, j] < threshold:
                temp[i, j] = 0
            elif distribution_map[i, j] > 1 - threshold: 
                temp[i, j] = 255
    return temp

def boxFilter(img_in ,size):
    img_fil = np.zeros( shape=(img_in.shape[0] , img_in.shape[1] ),dtype=int)
    row , col = img_fil.shape
    for i in range(row):
        for j in range(col):
            img_fil[i][j] = np.mean(img_in[i: i + size, j: j + size])
    return img_fil

def medianFilter(img_in ,size):
    img_fil = np.zeros( shape=(img_in.shape[0] , img_in.shape[1] ),dtype=int)
    row , col = img_fil.shape
    for i in range(row):
        for j in range(col):
            img_fil[i][j] = np.median(img_in[i: i + size, j: j + size])
    return img_fil
def normalize(img_in):
            row, col = img_in.shape
            img_nor = np.zeros(img_in.shape )
            ##max = np.max(img_in)
            ##min = np.min(img_in)
            for i in range(row):
                for j in range(col):
                    #img_nor[i,j] = ((img_in[i,j]- min)/(max-min))
                    img_nor[i,j] = img_in[i,j]/255
            return img_nor

def SNR(img , noisyimage):
    imgN ,noisyimageN = normalize(img) , normalize(noisyimage)
    mu1,mu2,power1,power2 =0,0,0,0   
    row, col = noisyimageN.shape
    for i in range(row):
        for j in range(col):
            mu1 = mu1 +imgN[i,j]
    mu1 = mu1/(row*col)    
    for i in range(row):
        for j in range(col):
            mu2 = mu2 + (noisyimageN[i, j] - imgN[i, j])
    mu2 = mu2 / (row * col)
   
    for i in range(row):
        for j in range(col):
            power1 = power1 +math.pow(imgN[i, j] - mu1, 2)
    power1 = power1/(row*col)
    
    for i in range(row):
        for j in range(col):
            power2 = power2 +  math.pow(noisyimageN[i, j] - imgN[i, j] - mu2, 2)
    power2 = power2 / (row * col)
    
    
    SNR = 20*(math.log10(math.sqrt(power1/power2)))
    return SNR

test = cv2.imread('test.bmp', 2)
print('snr test: ', SNR(img, test))
g10 = getGaussNoise(img, 0, 1, 10)
cv2.imwrite('lena_g10.bmp' , g10)
g30 = getGaussNoise(img, 0, 1, 30)
cv2.imwrite('lena_g30.bmp' , g30)
print('done gauss')

sp01 =getSaltPepperNoise(img, 0.1)
cv2.imwrite('lena_sp01.bmp' , sp01)
sp005 =getSaltPepperNoise(img, 0.05)
cv2.imwrite('lena_sp005.bmp' , sp005)
print('done salt pepper')

g10box3 = boxFilter(g10 ,(1))
cv2.imwrite('Parta_Lena_g10box3.bmp',g10box3)
g30box3 = boxFilter(g30 ,(3)) 
cv2.imwrite('Parta_Lena_g30box3.bmp',g30box3)
g10box5 = boxFilter(g10 ,(5))
cv2.imwrite('Parta_Lena_g10box5.bmp',g10box5)
g30box5 = boxFilter(g30 ,(5)) 
cv2.imwrite('Parta_Lena_g30box5.bmp',g30box5)
print('done gauss box')

sp01box3 = boxFilter(sp01, 3)
cv2.imwrite('Partb_Lena_sp01box3.bmp',sp01box3)
sp005box3 = boxFilter(sp005, 3)
cv2.imwrite('Partb_Lena_sp005box3.bmp',sp005box3)
sp01box5 = boxFilter(sp01, 5)
cv2.imwrite('Partb_Lena_sp01box5.bmp',sp01box5)
sp005box5 = boxFilter(sp005, 5)
cv2.imwrite('Partb_Lena_sp005box5.bmp',sp005box5)
print('done sp box')

g10median3 = medianFilter(g10, 3)
cv2.imwrite('Partc_Lena_g10median3.bmp', g10median3)
g30median3 = medianFilter(g30, 3)
cv2.imwrite('Partc_Lena_g30median3.bmp', g30median3)
g10median5 = medianFilter(g10, 5)
cv2.imwrite('Partc_Lena_g10median5.bmp', g10median5)
g30median5 = medianFilter(g30, 5)
cv2.imwrite('Partc_Lena_g30median5.bmp', g30median5)
print('done gauss median')

sp01median3 = medianFilter(sp01, 3)
cv2.imwrite('Partd_Lena_sp01median3.bmp', sp01median3)
sp005median3 = medianFilter(sp005, 3)
cv2.imwrite('Partd_Lena_sp005median3.bmp', sp005median3)
sp01median5 = medianFilter(sp01, 5)
cv2.imwrite('Partd_Lena_sp01median5.bmp', sp01median5)
sp005median5 = medianFilter(sp005, 5)
cv2.imwrite('Partd_Lena_sp005median5.bmp', sp005median5)
print('done sp median')


g10closeopen =opening(closing(g10, kernel), kernel)
cv2.imwrite('Parte1_Lena_g10co.bmp', g10closeopen)
g30closeopen =opening(closing(g30, kernel), kernel)
cv2.imwrite('Parte1_Lena_g30co.bmp', g30closeopen)
sp01closeopen =opening(closing(sp01, kernel), kernel)
cv2.imwrite('Parte1_Lena_sp01co.bmp', sp01closeopen)
sp005closeopen =opening(closing(sp005, kernel), kernel)
cv2.imwrite('Parte1_Lena_sp005co.bmp', sp005closeopen)
print('done closeopen')
g10openclose = closing(opening(g10, kernel), kernel)
cv2.imwrite('Parte2_Lena_g10oc.bmp', g10openclose)
g30openclose = closing(opening(g30, kernel), kernel)
cv2.imwrite('Parte2_Lena_g30oc.bmp', g30openclose)
sp01openclose = closing(opening(sp01, kernel), kernel)
cv2.imwrite('Parte2_Lena_sp01oc.bmp', sp01openclose)
sp005openclose = closing(opening(sp005, kernel), kernel)
cv2.imwrite('Parte2_Lena_sp005oc.bmp', sp005openclose)
print('done open close')

imglist = [g10, g30, sp01, sp005, g10box3, g30box3, g10box5, g30box5, sp01box3, sp005box3, sp01box5, sp005box5, 
           g10median3, g30median3, g10median5, g30median5, sp01median3, sp005median3, sp01median5, sp005median5,
           g10closeopen, g30closeopen, sp01closeopen, sp005closeopen, g10openclose, g30openclose, sp01openclose, sp005openclose]
for element in imglist:
    print(SNR(img, element))













    