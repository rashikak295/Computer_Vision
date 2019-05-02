import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import itertools
import random
from itertools import starmap

angles = np.deg2rad(np.arange(0.0, 360.0))
cos = np.cos(angles)
sin = np.sin(angles)
dmax = math.floor(math.sqrt(1200**2 + 1600**2)/2)

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(x_diff, y_diff)
    if div == 0:
        return None 
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y

def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection: 
                    intersections.append(intersection)
    return intersections

def find_vanishing_point(img, grid_size, intersections):
    image_height = img.shape[0]
    image_width = img.shape[1]
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1
    max_intersections = 0
    best_cell = (0.0, 0.0)
    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        current_intersections = 0  
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)       
    if best_cell[0] != None and best_cell[1] != None:
        rx1 = int(best_cell[0] - grid_size / 2)
        ry1 = int(best_cell[1] - grid_size / 2)
        rx2 = int(best_cell[0] + grid_size / 2)
        ry2 = int(best_cell[1] + grid_size / 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 10)
    return best_cell

def FindLines(a,x,y):
	point = [None]*dmax
	accumulator = np.zeros(dmax)
	for i in range(len(x)):	
		d = math.floor((y[i]-600)*sin[a]+(x[i]-800)*cos[a])
		if d > 0 :
			accumulator[d] += 1
			if point[d]!=None:
				point[d].append([x[i],y[i]])
			else: point[d] = [[x[i],y[i]]]			 
	threshold = 100
	peaks = np.where(np.r_[True, accumulator[1:] > accumulator[:-1]] & np.r_[accumulator[:-1] > accumulator[1:], True] & np.r_[accumulator[:-1] > threshold, True] != False)
	points = [point[x] for x in peaks[0]]
	return peaks[0], points

path = './ST2MainHall4'
files = ['ST2MainHall4012.jpg','ST2MainHall4040.jpg','ST2MainHall4066.jpg']
for name in files:
	print(name)
	image0 = cv2.imread('./ST2MainHall4/'+name)
	image = cv2.blur(image0,(5,5))
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image1 = image0.copy()
	image2 = image0.copy()
	image3 = image0.copy()
	image4 = image0.copy()
	image5 = image0.copy()
	image6 = image0.copy()
	edges = cv2.Canny(image,50,150,apertureSize = 3)
	y,x = np.nonzero(edges) 

	grx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
	gry = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
	magg, angleg = cv2.cartToPolar(grx, gry, angleInDegrees=True)
	angleg = np.where(angleg>=180,angleg-180,angleg) 
	angleg = np.where(angleg==180,0,angleg)
	histg, binsg = np.histogram(angleg,36)
	sig_angles = np.where(np.r_[True, histg[1:] > histg[:-1]] & np.r_[histg[:-1] > histg[1:], True] != False)
	sig_angles = sig_angles[0]*5
	greater = sig_angles.copy()
	greater = greater+180
	sig_angles = np.concatenate((greater,sig_angles))
	all_linesf = []
	for ang in sig_angles:
		for an in range(ang-3,ang+4):
			if an>=0:
				peaksh, pointsh = FindLines(an,x,y)
				for l in range(len(peaksh)):
					alh = cos[an]
					blh = sin[an]
					x0h = alh*peaksh[l]
					y0h = blh*peaksh[l]
					x1h = int(x0h + 1000*(-blh) + 800)
					y1h = int(y0h + 1000*(alh) + 600)
					x2h = int(x0h - 1000*(-blh) + 800)
					y2h = int(y0h - 1000*(alh) + 600)
					all_linesf.append([(x1h,y1h),(x2h,y2h)])
					cv2.line(image3,(x1h,y1h),(x2h,y2h),(0,0,255),1)
	cv2.imwrite('edge_histogram'+name,image3)

	if all_linesf:
		intersectionsf = find_intersections(all_linesf)
		if intersectionsf:
			grid_sizef = min(image4.shape[0], image4.shape[1])//20
			print("Vanishing points using Edge Histogram:")
			vanishing_point = find_vanishing_point(image6, grid_sizef, intersectionsf)
			print("Best cell:", vanishing_point)
			cv2.imwrite('histogram_vanishing_points'+name,image6)

	# for a in [random.sample(range(360),2)]: #For getting 2 random angles use this instead.
	for a in [45,180]:
		peaks, points = FindLines(a,x,y)
		for i in range(len(peaks)):
			al = cos[a]
			bl = sin[a]
			x0 = al*peaks[i]
			y0 = bl*peaks[i]
			x1 = int(x0 + 1000*(-bl) + 800)
			y1 = int(y0 + 1000*(al) + 600)
			x2 = int(x0 - 1000*(-bl) + 800)
			y2 = int(y0 - 1000*(al) + 600)
			cv2.line(image0,(x1,y1),(x2,y2),(0,0,255),1)
			for p in points[i]:
				image0[p[1],p[0]] = [0,255,0]
	cv2.imwrite('findlines'+name,image0)

	lines = cv2.HoughLines(edges,1,np.pi/180,100)
	all_linesh = []
	for line in lines:
		rho,theta = line[0]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 2000*(-b))
		y1 = int(y0 + 2000*(a))
		x2 = int(x0 - 2000*(-b))
		y2 = int(y0 - 2000*(a))
		all_linesh.append([(x1,y1),(x2,y2)])
		cv2.line(image1,(x1,y1),(x2,y2),(0,0,255),1)
	cv2.imwrite('houghlines'+name,image1)

	if all_linesh:
		intersections = find_intersections(all_linesh)
		if intersections:
			grid_size = min(image4.shape[0], image4.shape[1])//20
			print("Vanishing points using Hough Transform:")
			vanishing_point = find_vanishing_point(image4, grid_size, intersections)
			print("Best cell:", vanishing_point)
			cv2.imwrite('vanishing_points'+name,image4)

	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,1000,0)
	all_linesp = []
	for line in lines:	
		x1,y1,x2,y2 = line[0]
		all_linesp.append([(x1,y1),(x2,y2)])
		cv2.line(image2,(x1,y1),(x2,y2),(0,0,255),1)
	cv2.imwrite('prob_houghlines'+name,image2)

	if all_linesp:
		intersectionsp = find_intersections(all_linesp)
		if intersectionsp:
			grid_sizep = min(image5.shape[0], image5.shape[1])//20
			print("Vanishing points using Probabilistic Hough Transform:")
			vanishing_point = find_vanishing_point(image5, grid_sizep, intersectionsp)
			print("Best cell:", vanishing_point)
			cv2.imwrite('prob_vanishing_points'+name,image5)	