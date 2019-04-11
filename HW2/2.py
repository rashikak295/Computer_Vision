import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import itertools
from sklearn import preprocessing

path = './ST2MainHall4'
files = os.listdir(path)
histograms = []

print("Calculating Histograms...")
for name in files:
	image = cv2.imread('./ST2MainHall4/'+name)
	image = np.array(image, dtype=np.int16)
	new_image = ((image[:,:,2] >> 5) << 6) + ((image[:,:,1] >> 5) << 3) + (image[:,:,0] >> 5)
	hist, bins = np.histogram(new_image, 512,[0,512]) 
	histograms.append(hist)

plt.subplot(221), plt.plot(histograms[0]), plt.title('Scene1 Histogram'), plt.subplots_adjust(hspace=.4)
plt.subplot(222), plt.plot(histograms[30]), plt.title('Scene2 Histogram')
plt.subplot(223), plt.plot(histograms[60]), plt.title('Scene3 Histogram'), plt.subplots_adjust(wspace=.3)
plt.subplot(224), plt.plot(histograms[90]), plt.title('Scene4 Histogram'), plt.subplots_adjust(wspace=.3)
plt.show()

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

print("Calculating normalized Histogram Intersection and Chi-square...")
all_chi = np.zeros((99,99))
all_int = np.zeros((99,99))
for h1,h2 in itertools.combinations(range(len(histograms)), 2):
	all_chi[h1][h2] = chi_square(histograms[h1],histograms[h2])
	all_chi[h2][h1] = all_chi[h1][h2]
	all_int[h1][h2] = intersection(histograms[h1],histograms[h2])
	all_int[h2][h1] = all_int[h1][h2]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
normalized_all_chi = min_max_scaler.fit_transform(all_chi)
normalized_all_int = min_max_scaler.fit_transform(all_int)

plt.imshow(normalized_all_int), plt.title('Histogram Intersection')
plt.show()
plt.imshow(normalized_all_chi), plt.title('Chi-square')
plt.show()