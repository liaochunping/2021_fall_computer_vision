import cv2

def flipv(imgg):
    #img2= numpy.zeros([imgg.shape[0], imgg.shape[1],imgg.shape[2]])
    img2 = imgg
    for i in range(imgg.shape[0]):

        img2[i,:]=imgg[imgg.shape[0]-i-1,:]

    return img2

img = cv2.imread('lena.bmp')
image = cv2.imread('lena.bmp')
row,col,typ = img.shape

img2 = flipv(image)
#for i in range(0,row):
   # image[i] = img[row-i-1]
    
cv2.imshow('Lef', image)

    
for i in range(0,row//2):
    for j in range(col):
        img[i][j] , img[row-i-1][j]=img[row-i-1][j] , img[i][j]

cv2.imshow('vertical', img2)

cv2.imshow('Le', image)
#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()