import cv2
img = cv2.imread('lena.bmp')
image = cv2.imread('lena.bmp')
row,col,typ = img.shape

for i in range(0,row):
    for j in range(0,col):    
        image[i][j] = img[row-i-1][j]
    
cv2.imshow('vetical', image)
cv2.imwrite('vetical.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()