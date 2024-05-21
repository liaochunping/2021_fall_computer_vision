import cv2
img = cv2.imread('lena.bmp')
image = cv2.imread('lena.bmp')
row,col,typ = img.shape

for i in range(0,row):
    for j in range(0 , col):
        image[i][j] = img[i][col-j-1]
    
cv2.imshow('horizontal', image)
cv2.imwrite('horizontal.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()