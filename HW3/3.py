import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import itertools
from sklearn import preprocessing

def intersection(hist_1, hist_2):
	minima = np.minimum(hist_1, hist_2)
	maxima = np.maximum(hist_1, hist_2)
	intersection = np.true_divide(np.sum(minima), np.sum(maxima))
	return intersection

def chi_square(hist_1, hist_2):
	chi = 0.0
	for h1,h2 in zip(hist_1,hist_2):
		if h1+h2>=5:
			chi = chi + (((h1-h2)**2)/(h1+h2))
	return chi

path = './ST2MainHall4'
files = os.listdir(path)
gray_histogram = []
color_histogram = []
eigen_histogram = []

for name in files:
	print(name)
	image = cv2.imread('./ST2MainHall4/'+name)
	image = cv2.blur(image,(5,5))
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	grx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=3)
	gry = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=3)
	b = image[:,:,0]
	g = image[:,:,1]
	r = image[:,:,2]
	bx = cv2.Sobel(b,cv2.CV_64F,1,0,ksize=3)
	by = cv2.Sobel(b,cv2.CV_64F,0,1,ksize=3)
	gx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
	gy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
	rx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize=3)
	ry = cv2.Sobel(r,cv2.CV_64F,0,1,ksize=3)

	magg, angleg = cv2.cartToPolar(grx, gry, angleInDegrees=True)
	angleg = np.uint8(np.rint(np.divide(angleg,10)))
	histg, binsg = np.histogram(angleg,36)
	gray_histogram.append(histg)

	magc,anglec = cv2.cartToPolar(bx+gx+rx,by+gy+ry, angleInDegrees=True)
	anglec = np.uint8(np.rint(np.divide(anglec,10)))
	magc = abs(bx)+abs(gx)+abs(rx)+abs(by)+abs(gy)+abs(ry)
	magc[magc<5] = 0
	histc, binsc = np.histogram(anglec,36, weights = magc)
	color_histogram.append(histc)
	
	a = rx**2 + gx**2 + bx**2
	b = rx*ry + gx*gy + bx*by
	c = ry**2 + gy**2 + by**2
	lamda = 1/2*((a+c)+np.sqrt((a+c)**2 - 4*(b**2-a*c)))
	x = np.divide(-b,a-lamda)
	mage, anglee = cv2.cartToPolar(x, np.ones(x.shape), angleInDegrees=True)
	mage = lamda
	anglee = np.uint8(np.rint(np.divide(anglee,10)))
	histe, binse = np.histogram(anglee,36, weights = mage)
	eigen_histogram.append(histe)

	if name == "ST2MainHall4001.jpg":
		X, Y = np.meshgrid(np.arange(1600), np.arange(1200))
		plt.figure()
		plt.subplot(221), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(grx[::50,::50]), np.uint8(gry[::50,::50]), units='width'), plt.title('Scene 1 Gray Image Orientation')
		plt.subplot(222), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(bx[::50,::50]), np.uint8(by[::50,::50]), units='width', color='blue'), plt.title('Scene 1 Blue channel Image Orientation')
		plt.subplot(223), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(gx[::50,::50]), np.uint8(gy[::50,::50]), units='width', color='green'), plt.title('Scene 1 Green channel  Image Orientation') 
		plt.subplot(224), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(rx[::50,::50]), np.uint8(ry[::50,::50]), units='width', color='red'), plt.title('Scene 1 Red channel Image Orientation')
		plt.show()
		plt.subplot(231), plt.imshow(magg), plt.title('Scene 1 Gray Edge Magnitude'), plt.subplots_adjust(hspace=.4)
		plt.subplot(232), plt.imshow(magc), plt.title('Scene 1 Color Edge Magnitude'), plt.subplots_adjust(hspace=.4)
		plt.subplot(233), plt.imshow(mage), plt.title('Scene 1 Color Edge Magnitude using eigen values'), plt.subplots_adjust(hspace=.4)
		plt.subplot(234), plt.plot(histg), plt.title('Scene 1 Gray Edge Histogram'), plt.subplots_adjust(hspace=.4)
		plt.subplot(235), plt.plot(histc), plt.title('Scene 1 Color Edge Histogram'), plt.subplots_adjust(hspace=.4)
		plt.subplot(236), plt.plot(histe), plt.title('Scene 1 Color Edge Histogram using Eigen Values'), plt.subplots_adjust(hspace=.4)
		plt.show()
	if name=="ST2MainHall4031.jpg":
		X, Y = np.meshgrid(np.arange(1600), np.arange(1200))
		plt.figure()
		plt.subplot(221), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(grx[::50,::50]), np.uint8(gry[::50,::50]), units='width'), plt.title('Scene 2 Gray Image Orientation')
		plt.subplot(222), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(bx[::50,::50]), np.uint8(by[::50,::50]), units='width', color='blue'), plt.title('Scene 2 Blue channel Image Orientation')
		plt.subplot(223), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(gx[::50,::50]), np.uint8(gy[::50,::50]), units='width', color='green'), plt.title('Scene 2 Green channel  Image Orientation') 
		plt.subplot(224), plt.quiver(X[::50,::50], Y[::50,::50], np.uint8(rx[::50,::50]), np.uint8(ry[::50,::50]), units='width', color='red'), plt.title('Scene 2 Red channel Image Orientation')
		plt.show()
		plt.subplot(231), plt.imshow(magg), plt.title('Scene 2 Gray Edge Magnitude'), plt.subplots_adjust(hspace=.4)
		plt.subplot(232), plt.imshow(magc), plt.title('Scene 2 Color Edge Magnitude'), plt.subplots_adjust(hspace=.4)
		plt.subplot(233), plt.imshow(mage), plt.title('Scene 2 Color Edge Magnitude using eigen values'), plt.subplots_adjust(hspace=.4)
		plt.subplot(234), plt.plot(histg), plt.title('Scene 2 Gray Edge Histogram'), plt.subplots_adjust(hspace=.4)
		plt.subplot(235), plt.plot(histc), plt.title('Scene 2 Color Edge Histogram'), plt.subplots_adjust(hspace=.4)
		plt.subplot(236), plt.plot(histe), plt.title('Scene 2 Color Edge Histogram using Eigen Values'), plt.subplots_adjust(hspace=.4)
		plt.show()

allc_chi = np.zeros((99,99))
allc_int = np.zeros((99,99))
for h1,h2 in itertools.combinations(range(len(color_histogram)), 2):
	allc_chi[h1][h2] = chi_square(color_histogram[h1],color_histogram[h2])
	allc_chi[h2][h1] = allc_chi[h1][h2]
	allc_int[h1][h2] = intersection(color_histogram[h1],color_histogram[h2])
	allc_int[h2][h1] = allc_int[h1][h2]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
normalized_allc_chi = min_max_scaler.fit_transform(allc_chi)
normalized_allc_int = min_max_scaler.fit_transform(allc_int)

allg_chi = np.zeros((99,99))
allg_int = np.zeros((99,99))
for h1,h2 in itertools.combinations(range(len(gray_histogram)), 2):
	allg_chi[h1][h2] = chi_square(gray_histogram[h1],gray_histogram[h2])
	allg_chi[h2][h1] = allg_chi[h1][h2]
	allg_int[h1][h2] = intersection(gray_histogram[h1],gray_histogram[h2])
	allg_int[h2][h1] = allg_int[h1][h2]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
normalized_allg_chi = min_max_scaler.fit_transform(allg_chi)
normalized_allg_int = min_max_scaler.fit_transform(allg_int)

alle_chi = np.zeros((99,99))
alle_int = np.zeros((99,99))
for h1,h2 in itertools.combinations(range(len(eigen_histogram)), 2):
	alle_chi[h1][h2] = chi_square(eigen_histogram[h1],eigen_histogram[h2])
	alle_chi[h2][h1] = alle_chi[h1][h2]
	alle_int[h1][h2] = intersection(eigen_histogram[h1],eigen_histogram[h2])
	alle_int[h2][h1] = alle_int[h1][h2]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
normalized_alle_chi = min_max_scaler.fit_transform(alle_chi)
normalized_alle_int = min_max_scaler.fit_transform(alle_int)

plt.subplot(231), plt.imshow(normalized_allg_int), plt.title('Gray Image Histogram Intersection')
plt.subplot(232), plt.imshow(normalized_allc_int), plt.title('Color Image Histogram Intersection')
plt.subplot(233), plt.imshow(normalized_alle_int), plt.title('Color Image Histogram Intersection (Using Eigen values)')
plt.subplot(234), plt.imshow(normalized_allg_chi), plt.title('Gray Image Chi-square')
plt.subplot(235), plt.imshow(normalized_allc_chi), plt.title('Color Image Chi-square')
plt.subplot(236), plt.imshow(normalized_alle_chi), plt.title('Color Image Chi-square (Using Eigen values)')
plt.show()