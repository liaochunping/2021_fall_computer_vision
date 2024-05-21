import numpy as np
import cv2
import matplotlib.pyplot as plt
def img_binarize(img_in ):
    shape = img_in.shape
    binimg = np.zeros(shape )
    for i in range(shape[0]):
        for j in range(shape[1]):
           
                if img_in[i][j] >= 128:
                    binimg[i][j] =1
                else:
                    binimg[i][j] = 0
    return binimg


def img_downsample(img_in):
    #Downsampling Lena from 512x512 to 64x64,
    img_down = np.zeros((64,64), dtype =int)   
    for i in range(img_down.shape[0]):
        for j in range(img_down.shape[1]):
            img_down[i][j] = img_in[8*i][8*j]  
    return img_down

def getNeighbors(x,y ,img_in ):
    return [getValue(img_in, x-1, y), getValue(img_in, x-1, y+1), getValue(img_in, x, y+1),       \
			getValue(img_in, x+1,y+1), getValue(img_in, x+1,y), getValue(img_in, x+1, y-1),     \
			getValue(img_in, x,y-1), getValue(img_in, x-1,y-1)]

def getValue(img_in , x ,y):
    if x >=img_in.shape[0] or x <0 or y >= img_in.shape[1] or y <0:
        return 0
    return img_in[x][y]


def transitions(getNeibors):
    n = getNeibors +getNeibors[0:1]
    return sum((n1,n2) == (0,1)for n1 ,n2 in zip(n , n[1:]))


def thinning(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
    
        changing1 = []
        for x in range(1, len(image) - 1):
            for y in range(1, len(image[0]) - 1):
                P1,P2,P3,P4,P5,P6,P7,P8 = n = getNeighbors(x, y, image)
                if (image[x][y] == 1 and    
                    P3 * P5 * P7 == 0 and   
                    P1 * P3 * P5 == 0 and   
                    transitions(n) == 1 and 
                    2 <= sum(n) <= 6):      
                    changing1.append((x,y))
                    
        for x, y in changing1: 
            image[x][y] = 0
        
        changing2 = []
        for x in range(1, len(image) - 1):
            for y in range(1, len(image[0]) - 1):
                P1,P2,P3,P4,P5,P6,P7,P8 = n = getNeighbors(x, y, image)
                if (image[x][y] == 1 and    
                    P1 * P5 * P7 == 0 and  
                    P1 * P3 * P7 == 0 and   
                    transitions(n) == 1 and 
                    2 <= sum(n) <= 6):      
                    changing2.append((x,y))
                    
        for x, y in changing2:
            image[x][y] = 0
        
    return image
img = cv2.imread('lena.bmp', 2)
binimg = img_binarize(img)
imgdown = img_downsample(binimg)   
imgthin =thinning(imgdown)

plt.imshow(imgthin , cmap = plt.cm.gray)
plt.savefig('test.png', cmap = plt.cm.gray)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    