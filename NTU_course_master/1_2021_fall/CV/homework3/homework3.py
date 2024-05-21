import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_histogram(img_in):
    # a histogram
    xaxies = np.zeros(256  ,dtype = int)
    shape = img_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            xaxies[img_in[i][j]] +=1
    return xaxies
def dark_img(img_in):
    darkimg = np.zeros(img_in.shape)
    shape = img_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            darkimg[i][j] = (int(img_in[i][j]//3))
            
    darkimg = darkimg.astype(int)
    return darkimg
    
def equalization(img_in ,his_in):
    shape = his_in.shape
    pdf = np.zeros(shape[0])
    cdf = np.zeros(shape[0])
    img_eq = np.zeros(img_in.shape)
    
    label_after_equ =  np.zeros(shape[0])#數值標籤轉換
    after_equ =  np.zeros(shape[0])
    
    for i in range(shape[0]):
        pdf[i] = his_in[i]/np.sum(his_in)
    cdf = pdf.cumsum()
    for i in range(shape[0]):
        label_after_equ[i] =round(cdf[i] *255)
     #累積機率到每個像素值，並四捨五入
    for i in range(shape[0]):
        after_equ[label_after_equ[i].astype(int)] += his_in[i]
        
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            img_eq[i][j] = label_after_equ[img_in[i][j]]
   
    
    
    return after_equ , img_eq
        
    
    
    
colorimg = cv2.imread('lena.bmp', 2)
hist = img_histogram(colorimg)
darkimg = dark_img(colorimg)

dark_hist = img_histogram(darkimg)
after_equalization = equalization(darkimg, dark_hist)

plt.bar(range(0,256) , dark_hist)
plt.savefig('dark_histogram.png')
plt.show()

plt.bar(range(0, 256), hist)
plt.savefig('histogram.png')
plt.show()

plt.bar(range(0,256) , after_equalization[0])
plt.savefig('equlization_hist.png')
plt.show()
cv2.imwrite('lena_after_dark.png' , darkimg)
cv2.imwrite('lena_after_equal.png' ,after_equalization[1])

