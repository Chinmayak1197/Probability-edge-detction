#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
from skimage.transform import rotate
import matplotlib.pyplot as plt

#  Gaussian Filter 
def gaussian(filter_size, sigma_x,sigma_y,stddev_x,stddev_y):

	gauss = []
	for i in range(filter_size):
		for j in range(filter_size):
			
			x = i-sigma_x
			y = j - sigma_y
			gauss.append(-(((x)**2/(2*stddev_x**2))+((y)**2)/(2*stddev_y**2)))

	gauss = np.exp(np.array(gauss))*255
	output = gauss.reshape(filter_size,filter_size)
	return output

# Dog filter 
def DoG(orientation, Scale):
    
    # kernel : Sobel 
    s_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    s_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    filter_size = 59 
    
    sigma_x = filter_size//2
    sigma_y = filter_size//2

    # DoG filter 
    output = []
    for index in range(len(Scale)):
        gaussian_f = gaussian(filter_size,sigma_x,sigma_y,Scale[index],Scale[index])
        
        # cv2.filter2D(src, ddepth, kernel)
        G_x=cv2.filter2D(gaussian_f,-1,s_x)
        G_y=cv2.filter2D(gaussian_f,-1,s_y)

        for index in range(len(orientation)):
            dog=rotate(G_x,orientation[index])
            output.append(dog)

    return output


def LoG(scale): 

    # Kernel 	
    log_k = np.array([[0,1,0],[1,-4,1],[0,1,0]])

    filter_size = 39 
    sigma_x = filter_size//2
    sigma_y = filter_size//2

    # Filter: LoG
    output=[]
    for index in range(len(scale)):
        gaussian_f=gaussian(filter_size,sigma_x,sigma_y,scale[index],scale[index])
        log=cv2.filter2D(gaussian_f,-1,log_k)
        output.append(log)
    
    for index in range(len(scale)):
        gaussian_f=gaussian(filter_size,sigma_x,sigma_y,3*scale[index],3*scale[index])
        log=cv2.filter2D(gaussian_f,-1,log_k)
        output.append(log)
    
    return output

def lm_filter(orientations, scale):

	# kernel 
	s_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	s_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

	filter_size = 39 

	sigma_x = filter_size//2
	sigma_y = filter_size//2

	# Filter : LM 
	output=[]

	for index in range(len(scale[:-1])):
		for index_o in range(len(orientations)):
			gaussian_f=gaussian(filter_size,sigma_x,sigma_y,scale[index],3*scale[index])
			DoG_1=cv2.filter2D(gaussian_f,-1,s_x)
			DoG_1=rotate(DoG_1,orientations[index_o])

			output.append(DoG_1)
			DoG_2=cv2.filter2D(DoG_1,-1,s_x)
			DoG_2=rotate(DoG_2,orientations[index_o])
			output.append(DoG_2)


	LoG_f = LoG(scale)
	output.extend(LoG_f)

	for index in range(len(scale)):
		gaussian_f=gaussian(filter_size,sigma_x,sigma_y,scale[index],scale[index])
		output.append(gaussian_f)

	return output


# Reference : https://en.wiikipedia.org/wiki/Gabor_filter
def gabor(sigma, theta, Lambda, psi, gamma):

	sigmax = sigma 
	sigmay = float(sigma) / gamma

	n = 3
	x1 = n * sigmax * np.cos(theta)
	x2 = n * sigmay * np.sin(theta)
	maxx = max(abs(x1), abs(x2))
	xmax = np.ceil(max(1, maxx))

	y1 = n * sigmax * np.sin(theta)
	y2 = n * sigmay * np.cos(theta)
	maxy = max(abs(y1), abs(y2))
	ymax = np.ceil(max(1, maxy))

	minx , miny = -maxx , -maxy

	(y, x) = np.meshgrid(np.arange(miny, maxy + 1), np.arange(minx, maxx + 1))

	# angle
	thetay = -x * np.sin(theta) + y * np.cos(theta)
	thetax = x * np.cos(theta) + y * np.sin(theta)
	

	# Gabour filter 
	terme = thetax ** 2 / sigmax ** 2 + thetay ** 2 / sigmay ** 2
	ntheta = 2 * np.pi / Lambda * thetax + psi
	output = np.exp(-.5 * (terme)) * np.cos(ntheta)

	return output

def gabor_f(scales,thetas,Lambda,psi,gamma):
    
    # Gabour filter 
    output=[]
   
    for s in scales:
        for t in thetas:
            gabor_f=gabor(s,t,Lambda,psi,gamma)
            output.append(gabor_f)

    return output


def plotting(row, col, type, name, display):

	n = 0 
	_, array = plt.subplots(row,col,figsize=(15,15))

	for x in range(row):
		for y in range(col):
			array[x,y].imshow(type[n],cmap='gray')
			array[x,y].axis('off')
			
			n = n + 1
    
	plt.savefig(name)
	if(display):
		plt.show()

def halfdisk(scale, orientation):
    
    filter_size = (scale*2)+1
    image= np.zeros((filter_size,filter_size))
    
    center = [image.shape[0]//2,image.shape[1]//2]
    
    for x in range(filter_size):
        for y in range(filter_size//2):
            if(np.sqrt((x-center[0])**2+(y-center[1])**2) <= scale):
                image[x,y]=1

    return rotate(image,orientation)


def halfdiskmask(orientation,scale):
    output = []

    for i in scale:
        for j in orientation:

            output1=halfdisk(i,j)
            output.append(output1)
            
            output2=halfdisk(i,j +180)
            output.append(output2)
    
    return output  

from sklearn.cluster import KMeans

def textonmap(image,filters):

    outputs=[]

    heap = np.array(cv2.filter2D(image,-1,filters[0]))

    for i in filters[1:]:
        output=cv2.filter2D(image,-1,i)
        heap = np.dstack((heap,output))
        
    width,height,depth = heap.shape
    area = width * height
    heap=np.reshape(heap,(area,depth))

    # K-means clustering
    model=KMeans(n_clusters=64).fit(heap)
    outputs = np.reshape(model.predict(heap),(width,height))

    return outputs


def chisquare(mask1,mask2,image,number):

    image=image.astype(np.float32)
    summation =np.zeros(image.shape,dtype=np.float32)

    for i in range(number):

        copy=image.copy()
        copy[image==i]=1
        copy[image!=i]=0

        x=cv2.filter2D(copy,-1,mask1)
        y=cv2.filter2D(copy,-1,mask2)
        summation += (((x-y)**2)/(x+y+np.exp(-6)))
    
    return summation//2.0

def gradient(textronmap, halfdisk, number):


    distance = chisquare(halfdisk[0],halfdisk[0],textronmap,64)
    distance_heap = np.array(distance)
    n = len(halfdisk)

    for index in range(2,n//2,2):
        mask1=halfdisk[index]
        mask2=halfdisk[index+1]

        distance=chisquare(mask1,mask2,textronmap,number)
        distance_heap = np.dstack((distance_heap,distance))
    distance_heap = np.mean(distance_heap,axis=2)
    
    return distance_heap

# def brightnessmap(image):

#     grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#     width = grey.shape[0]
#     height = grey.shape[1]
#     area = width * height 
#     gray=np.reshape(grey,(area,1))

#     # K-means clustering 
#     model=KMeans(n_clusters=16).fit(grey)
#     output=np.reshape(model.predict(grey),(width,height))

#     return output

def brightnessmap(image):

    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    width,height = grey.shape
    grey=np.reshape(grey,(width*height,1))

    model=KMeans(n_clusters=16).fit(grey)
    pred=model.predict(grey)
    result = np.reshape(pred,(width,height))

    return result

def colormap(image):


    width,height,depth = image.shape

    image = np.reshape(image,(width*height,depth))

    model=KMeans(n_clusters=16).fit(image)

    pred=model.predict(image)
    result = np.reshape(pred,(width,height))

    return result


import os 

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	# Orientations between 0 and 360
	number = 8 
	change = int(360/number)
	orientation = []
	
	
	for x in range(0,360, change):
		orientation.append(x)

	# Scale
	Scale = [7,9]
	# DOG filter
	dog_f = DoG(orientation, Scale)
	# Printing DoG filter 
	plotting(2,8,dog_f,"DOG.png", False)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	small=[1,np.sqrt(2),2,2*np.sqrt(2)]
	large=[np.sqrt(2),2,2*np.sqrt(2),4]

	number = 6 
	change = int(360/number)
	orientation = []

	for x in range(0,360, change):
		orientation.append(x)

	lms_f = lm_filter(orientation, small)
	lml_f = lm_filter(orientation, large)

	plotting(6,8,lms_f,"LMS.png",False)
	plotting(6,8,lml_f,"LML.png",False)


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	scale = [3,5,7]

	psi, gamma = 1.0, 1.0  

	gabour_f = gabor_f(scale,np.linspace(0,90,5),8,psi,gamma)
	plotting(3,5,gabour_f,"Gabor.png",False)


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	orientation = [30,55,90,120]
	halfdiskm=halfdiskmask(orientation, [5,7,9,11])
	plotting(4,8,halfdiskm,"HDMasks.png",False)


	displaymaps=False
	displaygradients=False
	displayoutput=False

	print("Starting.......................")

	path = "../BSDS500/Images/"
	names = []
	
	for root,dirs,files in os.walk(path):
		for file in files:
			names.append(file.replace('.jpg',''))


	for image in names:
		print("Number ->",image)

		image_path=path+image+'.jpg'
		img=cv2.imread(image_path)


		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		combine_f=dog_f+lml_f+lms_f+gabour_f
	
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton_m=textonmap(img,combine_f)
		plt.imsave('./TextonMap_'+image+'.png',texton_m)
		if(displaymaps):
			plt.imshow(texton_m)
			plt.show()


		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		textrongradient=gradient(texton_m,halfdiskm,64)
		plt.imsave('./Tg'+image+'.png',textrongradient)
		if(displaygradients):
			plt.imshow(textrongradient)
			plt.show()


		"""
		Generate Brightness Map
		Perform brightness binning 
		"""


		brightness_m=brightnessmap(img)
		plt.imsave('./BrightnessMap'+image+'.png',brightness_m,cmap='binary')
		if(displaymaps):
			plt.imshow(brightness_m)
			plt.show()
		


		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightnessgradient=gradient(brightness_m,halfdiskm,16)
		plt.imsave('./Bg'+image+'.png',brightnessgradient,cmap='binary')
		if(displaygradients):
			plt.imshow(brightnessgradient)
			plt.show()
	

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		colorm=colormap(img)
		plt.imshow(colorm)
		plt.imsave('./ColorMap'+image+'.png',colorm)
		if(displaymaps):
			plt.imshow(colorm)
			plt.show()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		colorgradient=gradient(colorm,halfdiskm,16)
		plt.imsave('./Cg'+image+'.png',colorgradient)
		if(displaygradients):
			plt.imshow(colorgradient)
			plt.show()


		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobelpath="../BSDS500/SobelBaseline/"
		sobel=cv2.imread(sobelpath+image+'.png',0)
		
		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""

		cannypath="../BSDS500/CannyBaseline/"
		canny=cv2.imread(cannypath+image+'.png',0)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		totalgradient=(textrongradient+brightnessgradient+colorgradient)//3 
		

		total=0.5*sobel+0.5*canny
		pblite=np.multiply(totalgradient,total)
		
		plt.imsave('./PbLite'+image+'.png',pblite,cmap='gray')
		if(displayoutput):
			plt.imshow(pblite,cmap='gray')
			plt.show()

    
if __name__ == '__main__':
    main()
 


