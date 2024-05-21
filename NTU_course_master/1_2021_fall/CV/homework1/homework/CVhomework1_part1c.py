import cv2
img = cv2.imread('lena.bmp')
image = cv2.imread('lena.bmp')
row,col,typ = img.shape

for i in range(0,row):
    for j in range(0 , col):
        image[i][j] = img[col-j-1][row-i-1]
    
cv2.imshow('diagonal flip', image)
cv2.imwrite('diagonal flip.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 