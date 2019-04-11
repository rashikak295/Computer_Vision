import cv2
import numpy as np

image = cv2.imread('face.jpg',0) 
rows,cols = image.shape
blur = cv2.blur(image,(5,5)) 
ret,threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
transform = cv2.warpAffine(image,M,(cols,rows))
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(image,kernel,iterations = 1)
M2 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
rotation = cv2.warpAffine(image,M2,(cols,rows))

cv2.imwrite('blur.jpg',blur)
cv2.imwrite('threshold.jpg',threshold)
cv2.imwrite('rotation.jpg',rotation)
cv2.imwrite('transform.jpg',transform)
cv2.imwrite('erosion.jpg',erosion)

cv2.imshow('Blurred image',cv2.resize(blur,(300,400)))
cv2.imshow('Binary threshold',cv2.resize(threshold,(300,400)))
cv2.imshow('Rotation',cv2.resize(rotation,(300,400)))
cv2.imshow('Affine Transform',cv2.resize(transform,(300,400)))
cv2.imshow('Erosion',cv2.resize(erosion,(300,400)))
cv2.imshow('Original image',cv2.resize(image,(300,400)))
cv2.waitKey(0)
cv2.destroyAllWindows()