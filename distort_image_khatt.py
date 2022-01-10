# -*- coding: utf-8 -*-
import os
import codecs
import cv2
import numpy as np 
import sys
import shutil
from PIL import Image
import imageio
import matplotlib.pyplot  as plt
import random
from shutil import copyfile	
import glob
os.environ["PYTHONIOENCODING"] = "utf-8"
from PIL import Image,ImageFilter


f1=codecs.open('logDist.txt','a+','utf-8')
def preprocess2(v):
    
    
	backgrounds = os.listdir('/home/ubuntu/Sana/handwritten-text-recognition/src/backgroundIAM/')
	ch=random.choice(backgrounds)
 
	from PIL import Image
	img = Image.open('/home/ubuntu/Sana/handwritten-text-recognition/src/backgroundIAM/'+ch).convert('L')
	u=random.randint(1,4)
	if u == 4:
		img = img.transpose(Image.FLIP_TOP_BOTTOM)
	if u == 1:
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
	if u == 3:
		img = img.transpose(Image.ROTATE_90)
	if u == 2:
		img = img.transpose(Image.ROTATE_180)	
	
	
	widthb, heightb = img.size

	
	cv2.imwrite('a.png',v)
	img2 = Image.open('a.png').convert('L')
	
	widthl, heightl = img2.size
	#if widthl>widthb or heightl>heightb:
		#img=img.resize((widthl+100,heightl+100), Image.ANTIALIAS)
	img.save('output_file.jpg')
	
	
	img2.save('b.jpg')
	a=plt.imread('b.jpg',0)
	
	
	bg = plt.imread('output_file.jpg',0)
 
	size_a =  a.shape[1]*2 
	size_bg = bg.shape[1]
	
	o = 1
 
	while size_a > size_bg  :	
		bg  = np.concatenate((bg, bg), axis=1)
		size_bg=size_bg*2	
		 
 
 
	size_a1 =  a.shape[0] *2
	size_bg1 = bg.shape[0]
	
	o = 1
 
	while size_a1 > size_bg1  :	
		bg  = np.concatenate((bg, bg), axis=0)
		size_bg1=size_bg1*2	
 
	print('ground :', bg.shape)
	print('line: ', a.shape)
	 
		
	p = random.randint(1,100)
	p2 = random.randint(1,50)
		
			
	bg = bg[p:p+a.shape[0],p2:p2+a.shape[1]]



	param1 = random.randint(3,7)/10

	param2 = random.randint(3,7)/10
	
	#n=random.randint(10,90)
	a = cv2.addWeighted(bg,param1,a,param2,random.randint(-30,1))
	return a

 
def read_file(list_file_path):
    char_file = codecs.open(list_file_path, 'r', 'utf-8')

    list = []
    for l in char_file:
        list.append(l.strip())
    return list

def blur_image_low(img):

	kernel=random.randint(1,5)

	avging = cv2.blur(img,(kernel,kernel), cv2.BORDER_DEFAULT) 
	return avging	
 
def blur_image_hight(img):

	kernel=random.randint(6,15)

	avging = cv2.blur(img,(kernel,kernel), cv2.BORDER_DEFAULT) 
	return avging

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r
 
def erodecv(img):
	 
	  
	# Taking a matrix of size 5 as the kernel 
	k=random.randint(2,4)
	kernel = np.ones((k,k), np.uint8) 
	  
	# The first parameter is the original image, 
	# kernel is the matrix with which image is  
	# convolved and third parameter is the number  
	# of iterations, which will determine how much  
	# you want to erode/dilate a given image.  
	img_erosion = cv2.erode(img, kernel, iterations=1) 
	return img_erosion
def dilatecv(img):
	 
	  
	# Taking a matrix of size 5 as the kernel 
	k=random.randint(2,3)
	kernel = np.ones((k,k), np.uint8) 
	  
	# The first parameter is the original image, 
	# kernel is the matrix with which image is  
	# convolved and third parameter is the number  
	# of iterations, which will determine how much  
	# you want to erode/dilate a given image.  
	img_erosion = cv2.dilate(img, kernel, iterations=1) 
	return img_erosion


def distort_line(image):
	# Compute histogram
	im = image
	im = 255 - im



	thik1 = random.randint(2, 8)
	thik2 = random.randint(2, 5)
	thik3 = random.randint(5, 10)
	thik4 = random.randint(2, 10)
	newimage = image.copy()
	index_of_highest_peak=random.randint(20, 40)
	ind1 = index_of_highest_peak
	ind2 = index_of_highest_peak + random.randint(40, 50)
	ind3 = index_of_highest_peak - random.randint(40, 50)
	image_widh=im.shape[1]
 
	i1=random.randint(10, image_widh-5)
	i2 = random.randint(40,image_widh-20 )
	i3 = random.randint(10, image_widh-30)
	i4=random.randint(5, image_widh-10)

	cv2.line(newimage,  pt1=(i1,0), pt2=(i1,400), color=(0, 0, 0), thickness=thik1)
	cv2.line(newimage, pt1=(i3,0), pt2=(i3,400), color=(0, 0, 0), thickness=thik2)
	cv2.line(newimage,  pt1=(i2,0), pt2=(i2,400), color=(0, 0, 0), thickness=thik3)
	cv2.line(newimage, pt1=(i4, 0), pt2=(i4, 400), color=(0, 0, 0), thickness=thik4)
	
	
	return newimage


def distortion(set):
	i=0
	#ww='/home/ubuntu/Sana/BaseIAM/DatasetIAM/distorted/'
	#gt='/home/ubuntu/Sana/BaseIAM/DatasetIAM/GT_B/'
	#gt='/home/ubuntu/Sana/BaseIAM/DatasetIAM/GT_B/'
	gt='/home/ahmed/Desktop/Gan-OCR/Dataset/KHATT/Gt/Images/'
	ww = '/home/ubuntu/Sana/Hito-docs/DatasetKHATT1_hard3/'
	listf=read_file(set)
	#random.shuffle(listf)
	nbfiles=len(listf)
	sp1=nbfiles/8
	sp2=2*nbfiles/8
	sp3=3*nbfiles/8
	sp4 = 4*nbfiles/8
	sp5 = 5 * nbfiles / 8
	sp6 = 6 * nbfiles / 8
	sp7 = 7 * nbfiles / 8
	sp8 = 8 * nbfiles / 8

	print(len(listf))
	print(len(listf))
	for filename in  listf:
		 
			print(str(i))

			print(filename)
			
			#impath = '/home/ubuntu/Sana/BaseIAM/DatasetIAM/GT_B/' + filename +'.png'
			##binarize image GT
			 
			#gray = cv2.imread(impath)
			#gray=cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
			#ret,Binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
			#imageio.imwrite('/home/ubuntu/Sana/BaseIAM/DatasetIAM/GT_B/'+ filename+'.png',Binary)
			
			a = plt.imread(gt+ filename+'.tif')
			#a = plt.imread('E:\\Travail\\work\\dibco\\ResultsEvaluations\\IAM\\GT_B\\' + filename + '.png')

			plt.imsave('imagex.jpg',a,cmap='gray')
			a=plt.imread('imagex.jpg')
			im1=a
			im2=a
			im3=a
			#imageio.imwrite('/home/ubuntu/Sana/BaseIAM/GT/'+ filename,a)
			#########add background
			if i <sp1: 
				# im1=distort_line(im1)
				
				# im1=dilatecv(im1)
				# im1=blur_image_low(im1)
				# im1=preprocess2(im1)
				# imageio.imwrite(ww + filename+'.png',preprocess2(im1))
				# f1.write(filename + ' dilate,blur low,2 preprocess'+ '\n')
				##add blur
				im2=dilatecv(im2)
				bluredl=blur_image_hight(im2)
				f1.write(filename + ' dilate,blur highest,2 preprocess'+ '\n')
				imageio.imwrite(ww + filename+'.tif',preprocess2(bluredl))				
				
			# ########add low blur and background
			elif i >=sp1 and i <sp2:
				##add blur
				im2=dilatecv(im2)
				bluredl=blur_image_hight(im2)
				f1.write(filename + ' dilate,blur highest,2 preprocess'+ '\n')
				imageio.imwrite(ww + filename+'.tif',preprocess2(bluredl))
			elif i >= sp1 and i < sp2:
				##add blur
				im3=dilatecv(im3)
				bluredh=blur_image_hight(im3)
				bluredh=distort_line(bluredh)
				f1.write(filename + ' dilate,blur highest,line, preprocess'+ '\n')
				imageio.imwrite(ww + filename+'.tif',preprocess2(bluredh))
			elif i >= sp2 and i < sp3:
				x=dilatecv(im1)
				f1.write(filename + ' dilate, preprocess'+ '\n')
				imageio.imwrite(ww  + filename + '.tif',	preprocess2(x))
			elif i >= sp3 and i < sp4:
				
				x=preprocess2(im1)
				x=dilatecv(x)
				f1.write(filename + ' dilate preprocess'+ '\n'),
				imageio.imwrite(ww  + filename + '.tif',	preprocess2(x))
			elif i >= sp4 and i < sp5:
				x=dilatecv(im1)
				f1.write(filename + ' dilate,blur highest,preprocess'+ '\n')
				bluredl = blur_image_hight(x)
				imageio.imwrite(ww  + filename + '.tif',	preprocess2(bluredl))
			elif i >= sp5 and i < sp6:
				x=erodecv(im1)
				gauss = distort_line(x)
				bluredl = blur_image_low(gauss)
				f1.write(filename + ' erode,blur low,line, preprocess'+ '\n')
				imageio.imwrite(ww + filename + '.tif',	preprocess2(bluredl))
			elif i >= sp6 and i < sp7:
				bluredl=erodecv(im1)
		
				 
 
				f1.write(filename + ' erode, preprocess'+ '\n')
				imageio.imwrite(ww  + filename + '.tif',	preprocess2(bluredl))

				
			else :
				##add blur
				im2=dilatecv(im2)
				bluredl=blur_image_hight(im2)
				f1.write(filename + ' dilate,blur highest,2 preprocess'+ '\n')
				imageio.imwrite(ww + filename+'.tif',preprocess2(bluredl))
				
 
			i=i+1
 
			
if __name__ == '__main__':
 
	distortion('Sets/list_test')
	

