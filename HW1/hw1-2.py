import cv2
import numpy as np

image = cv2.imread('face.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg',gray)

cv2.imshow('Original image',cv2.resize(image,(300,400)))
cv2.imshow('Grayscale image',cv2.resize(gray,(300,400)))
cv2.waitKey(0)
cv2.destroyAllWindows()