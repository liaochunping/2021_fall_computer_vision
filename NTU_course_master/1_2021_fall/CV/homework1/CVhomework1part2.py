import cv2

def fliph(imgg):
    #img2= numpy.zeros([imgg.shape[0], imgg.shape[1],imgg.shape[2]])
    img2 = imgg
    for i in range(imgg.shape[1]):

        img2[i ,:]=imgg[imgg.shape[1]-i-1,]

    return img2

image = cv2.imread('lena.bmp')

#for i in range(0,row):
   # image[i] = img[row-i-1]
    
#cv2.imshow('Lef', image)

img2 = fliph(image)
cv2.imshow('vertical', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
