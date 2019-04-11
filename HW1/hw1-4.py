import cv2
import numpy as np

image = cv2.imread('face.jpg')
height, width, channels = image.shape 
A = cv2.pyrDown(image)
hei, wid, cha = A.shape 
images = [image, A]
while (hei>1 or wid>1): 
	A = cv2.pyrDown(A)
	hei, wid, cha = A.shape 
	images.append(A)
new_image = np.zeros((height,1353,3),dtype=np.uint8)
x_offset = 0
for i in range(2):
	h, w, c = images[i].shape
	new_image[:h,x_offset:x_offset + w,:] = images[i]
	x_offset = x_offset + w

x_offset = width
y_offset = images[1].shape[0]
for i in range(2,len(images)):	
	h, w, c = images[i].shape
	new_image[y_offset:y_offset + h,x_offset:x_offset + w,:] = images[i]
	x_offset = x_offset + w
	

cv2.imwrite('gaussian.jpg',new_image)
cv2.imshow('Original image',cv2.resize(image,(300,400)))
cv2.imshow('Gaussian Pyramid',cv2.resize(new_image,(450,400)))
cv2.waitKey(0)
cv2.destroyAllWindows()