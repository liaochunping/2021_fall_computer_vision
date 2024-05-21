import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_binarize(img_in ):
    #(a)a binary image (threshold at 128)
    shape = img_in.shape
    binimg = np.zeros(shape )
    for i in range(shape[0]):
        for j in range(shape[1]):
           
                if img_in[i][j] >= 128:
                    binimg[i][j] =255
                else:
                    binimg[i][j] = 0
    return binimg

def img_histogram(img_in):
    # (b) a histogram
    xaxies = np.zeros(256  ,dtype = int)
    shape = img_in.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            xaxies[img_in[i][j]] +=1
    
    return xaxies
 
colorimg = cv2.imread('lena.bmp', 2)
shape = colorimg.shape

LABEL_THRESHOLD = 500
cv2.imshow('test' , colorimg)
binimg = img_binarize(colorimg)
cv2.imshow('bin ', binimg)
cv2.imwrite('lenabin.png',binimg)
hist = img_histogram(colorimg)
plt.bar(range(0, 256), hist)
plt.savefig('histogram.png')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()