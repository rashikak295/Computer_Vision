import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
import unicodedata

mouseX, mouseY= 0,0
 
Tk().withdraw()
filename = askopenfilename()
image = cv2.imread(filename)

b = cv2.calcHist([image],[0],None,[256],[0,256])
g = cv2.calcHist([image],[1],None,[256],[0,256])
r = cv2.calcHist([image],[2],None,[256],[0,256])
plt.subplot(221), plt.plot(r,color = 'r'), plt.xlim([0,256]), plt.title('Histogram for red channel'), plt.subplots_adjust(hspace=.4)
plt.subplot(222), plt.plot(g,color = 'g'), plt.xlim([0,256]), plt.title('Histogram for green channel')
plt.subplot(223), plt.plot(b,color = 'b'), plt.xlim([0,256]), plt.title('Histogram for blue channel'), plt.subplots_adjust(wspace=.3)

print("\nClick anywhere on the image to get current pixel location and its details.")
print("Press ESC to quit.\n")
def click(event, x, y, flags, param):
	global mouseX, mouseY, image, clone
	iv = 0.0
	if event == cv2.EVENT_MOUSEMOVE:
		image = clone.copy()
		cv2.rectangle(image, (x-6,y-6), (x+6,y+6), (0, 255, 0), 1)
		mouseX, mouseY = x,y
		mean, sd = cv2.meanStdDev(image[(y-5):(y+6),(x-5):(x+6)])
		iv= (int(image[mouseY][mouseX][2])+int(image[mouseY][mouseX][1])+int(image[mouseY][mouseX][0]))/3.0
		print("(x="+str(mouseX)+", y="+str(mouseY)+") ~ R:"+str(image[mouseY][mouseX][2])+" G:"+str(image[mouseY][mouseX][1])+" B:"+str(image[mouseY][mouseX][0])+" \nIntensity value="+str(iv))
		print("Mean= R:"+str(mean[2][0])+" G:"+str(mean[1][0])+" B:"+str(mean[0][0]))
		print("Standard deviation= R:"+str(sd[2][0])+" G:"+str(sd[1][0])+" B:"+str(sd[0][0])+"\n")

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)
while True:
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
cv2.destroyAllWindows()
plt.show()