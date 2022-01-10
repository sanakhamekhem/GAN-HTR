import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import load_model
import imageio
import math
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
import random
import sys
import codecs
import re
import cv2
import tqdm
from glob import glob
from tqdm import tqdm
from data import preproc as pp


##########################################################################################################
##########################################################################################################
##########################################################################################################
rootPath='/content/drive/MyDrive/workIAM/'
DatabasePath='/content/drive/MyDrive/workIAM/TL_DIBCO'
scenario='S2_W10_IAM'

# define parameters
source = "khatt"
arch = "flor" ########ne pas modifier, nous utilisons architeture crnn de flor
batch_size=32
# define paths
source_path = os.path.join("..", "data", f"{source}.hdf5")
output_path = os.path.join("..", "output-crnn-gan-" + scenario  , source, arch)
target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
os.makedirs(output_path, exist_ok=True)

source_path2 = os.path.join("..", "data", f"{source}.hdf5")
output_path2 = os.path.join("..", "output-crnn-gan-progressive-" + scenario, source, arch)
target_path2 = os.path.join(output_path2, "checkpoint_weights.hdf5")
os.makedirs(output_path2, exist_ok=True)


# define input size, number max of chars per line and list of valid chars 
max_text_length = 128  ####not change this value
img_width=1024 #########for crnn
img_height=128 #########for crnn
input_size_crnn = (1024,128, 1)
input_size = (128,1024, 1) #############for the GAN
i =1 
flag = 0


def preproc(impath):
    # Load image, create blank mask, convert to grayscale, Gaussian blur
  # then adaptive threshold to obtain a binary image
  img = cv2.imread(impath)
  #== Parameters =======================================================================
  b,g,r = cv2.split(img)           # get b,g,r
  rgb_img = cv2.merge([r,g,b])     # switch it to rgb

  # Denoising
  
  #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,8)
  dst = cv2.fastNlMeansDenoisingColored(img,None,12,12,1,8)
  b,g,r = cv2.split(dst)           # get b,g,r
  rgb_dst = cv2.merge([r,g,b])     # switch it to rgb

  # save resulting masked image
  cv2.imwrite('result.png', rgb_dst)

  return rgb_dst
##########################################################################################################
##########################################################################################################
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def read_file(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())

	return list0
def read_file_char(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())

	return list0
 

def unet(pretrained_weights=None, input_size=(128, 1024, 1)):
	inputs = Input(input_size)

	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	bn = BatchNormalization(momentum=0.8)(conv1)
	bn.trainable=False
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv1)
	bn.trainable=False
	pool1 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	bn = BatchNormalization(momentum=0.8)(conv2)
	bn.trainable=False
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv2)
	bn.trainable=False
	pool2 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	bn = BatchNormalization(momentum=0.8)(conv3)
	bn.trainable=False
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv3)
	bn.trainable=False
	pool3 = MaxPooling2D(pool_size=(2, 2))(bn)

	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	bn = BatchNormalization(momentum=0.8)(conv4)
	bn.trainable=False
	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv4)
	bn.trainable=False
	drop4 = Dropout(0.5)(bn)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	bn = BatchNormalization(momentum=0.8)(conv5)
	bn.trainable=False
	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv5)
	bn.trainable=False
	drop5 = Dropout(0.5)(bn)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(drop5))
	# 	 merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up6)
	bn.trainable=False
	merge6 = concatenate([drop4, bn])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	bn = BatchNormalization(momentum=0.8)(conv6)
	bn.trainable=False
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv6)
	bn.trainable=False

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up7)
	bn.trainable=False
	merge7 = concatenate([conv3, bn])
	# 	 merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	bn = BatchNormalization(momentum=0.8)(conv7)
	bn.trainable=False
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv7)
	bn.trainable=False

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up8)
	bn.trainable=False
	merge8 = concatenate([conv2, bn])
	# 	 merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	bn = BatchNormalization(momentum=0.8)(conv8)
	bn.trainable=False
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv8)
	bn.trainable=False

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(bn))
	# 	 merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up9)
	bn.trainable=False
	merge9 = concatenate([conv1, bn])

	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	bn = BatchNormalization(momentum=0.8)(conv9)
	bn.trainable=False
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	bn.trainable=False
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	bn.trainable=False
	conv10 = Conv2D(1, 1, activation='sigmoid')(bn)

	model = Model(inputs=inputs, outputs=conv10)

	# 	 model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	return model
def unetorigine(pretrained_weights=None, input_size=(128,1024, 1)):
	inputs = Input(input_size)


	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	bn = BatchNormalization(momentum=0.8)(conv1)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(bn)


	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	bn = BatchNormalization(momentum=0.8)(conv2)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(bn)


	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	bn = BatchNormalization(momentum=0.8)(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(bn)


	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	bn = BatchNormalization(momentum=0.8)(conv4)
	conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv4)
	drop4 = Dropout(0.5)(bn)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	bn = BatchNormalization(momentum=0.8)(conv5)
	conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv5)
	drop5 = Dropout(0.5)(bn)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
# 	 merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up6)
	merge6 = concatenate ([drop4, bn])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	bn = BatchNormalization(momentum=0.8)(conv6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv6)


	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up7)
	merge7 = concatenate ([conv3, bn])
# 	 merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	bn = BatchNormalization(momentum=0.8)(conv7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv7)



	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
	bn = BatchNormalization(momentum=0.8)(up8)
	merge8 = concatenate ([conv2, bn])
# 	 merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	bn = BatchNormalization(momentum=0.8)(conv8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv8)


	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bn))
# 	 merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	bn = BatchNormalization(momentum=0.8)(up9)
	merge9 = concatenate ([conv1, bn])

	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
	bn = BatchNormalization(momentum=0.8)(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(bn)

	model = Model(inputs=inputs, outputs=conv10)

# 	 model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	return model

def get_optimizer():
	return Adam(lr=1e-4)

def vconcat_resize(img_list, interpolation
=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)



def split2(dataset,size,h,w):
    newdataset=[]
    nsize1=128
    nsize2=1024
    for i in range (size):
        im=dataset[i]
        for ii in range(0,h,nsize1): #2048
            for iii in range(0,w,nsize2): #1536
                newdataset.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
    return np.array(newdataset) 
def merge_image2(splitted_images, h,w):
    image=np.zeros(((h,w,1)))
    nsize1=128
    nsize2=1024
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
            ind=ind+1
    return np.array(image) 

def readGrayPair(deg_image_path):


  
  original_image = Image.open(deg_image_path)

  grey_image = original_image.convert('L')

  grey_image.save("deg_imageqq.tif")
  deg_image = plt.imread("deg_imageqq.tif")
 
  return deg_image 
def predict_gan_dibco(generator_path,dest):
	
	DatabasePathDistorted='handwritten-text-recognition/datasetsDIBCO/2010/H_DIBCO2010_test_images/'

	generator = unet()
	generator.load_weights(generator_path)

	count_image=1
	b=0
	for i in range(1,11):
		ext='.tiff'
		if i == 1 or i==3 or i==6:
			ext='.tif'
		elif i == 4:
			ext='.bmp'	
		else:
			ext='.jpg'
		noisy_image_path=DatabasePathDistorted + '/H' + str(i).zfill(2) + ext
		#preproc(noisy_image_path)
		original_image = Image.open(noisy_image_path)
		##############end padding
		#original_image = Image.open(noisy_image_path)  

		grey_image = original_image.convert('L')
		grey_image.save("deg_imagexw.png")

		test_image = plt.imread("deg_imagexw.png")



		h =  ((test_image.shape [0] // 128) +1)*128 
		w =  ((test_image.shape [1] // 1024 ) +1)*1024

		test_padding=np.zeros((h,w))+1
		test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

		test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
		predicted_list=[]
		n=0
		for l in range(test_image_p.shape[0]):

			print(str(i).zfill(2))
			i2=test_image_p[l]#*255
			i2=i2.astype('float64')
			imageio.imwrite('imbn.png', i2)
			deg_image =readGrayPair('imbn.png')
			pr=generator.predict(deg_image.reshape(1,128,1024,1))
			#plt.imsave('/content/drive/MyDrive/workIAM/TL_DIBCO/chunks/predicted/'+ str(b)+ '.png',pr.reshape(128, 1024),cmap='gray')
			#plt.imsave('/content/drive/MyDrive/workIAM/TL_DIBCO/chunks/distorted/'+ str(b)+ '.png',deg_image,cmap='gray')
			predicted_list.append(pr)

			#imageio.imwrite('/content/drive/MyDrive/workIAM/TL_DIBCO/chunks/GT/'+ str(b)+ '.png',i1)



			n=n+1
			b=b+1
		predicted_image = np.array(predicted_list)#.reshape()
		predicted_image=merge_image2(predicted_image,h,w)

		predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
		predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])
		sauv=predicted_image[:,:]
		predicted_image = (predicted_image[:,:])#*255
		predicted_image_float=predicted_image
		predicted_image=predicted_image.astype('float64')
		#print(predicted_image)
		os.makedirs('handwritten-text-recognition/src/' + dest, exist_ok=True)
		task='binarize'

		plt.imsave('handwritten-text-recognition/src/'+ dest + '/predicted_'+str(i)+'.png', predicted_image.reshape(
		  predicted_image.shape[0],predicted_image.shape[1]), cmap='gray')
		if task == 'binarize':

			bin_thresh = 0.0
			gray=sauv
			Binary = (gray[:,:]>bin_thresh)*1

			imageio.imwrite('handwritten-text-recognition/src/'+dest+'/b_predicted_'+str(i)+'.png',Binary)


		i=i+1
		count_image=count_image+1
def get_psnr(dest):
	
  DatabasePathGT='handwritten-text-recognition/datasetsDIBCO/2010/H_DIBCO2010_GT'
    
  count_image=1
  qo=0
  recap=0
  for i in range(1,11):	
      original_path_image_gt=DatabasePathGT + '/H' + str(i).zfill(2) + '.tiff'
      original_image = Image.open(original_path_image_gt)
     
      grey_image = original_image.convert('1')
      grey_image.save("gta.png")
      gt = plt.imread("gta.png")

      enhanced_image_path='handwritten-text-recognition/src/'+dest+'/b_predicted_' + str(i) + '.png'
      im2=Image.open(enhanced_image_path)
 
      im2=im2.convert('1')
      im2.save('im2a.png')
      predicted = plt.imread('im2a.png')

      
      psnrv=psnr(predicted,gt)
      print(psnrv)
      recap=recap+psnrv
      qo+=1
  av=recap/qo
  print('average psnr epoch 1  ')
  print(av)

def resul_tuning(generator_path,dest):

  
  predict_gan_dibco(generator_path,dest)
  get_psnr(dest)

if __name__ == '__main__':

	generator_path="weights/generator_weights.h5"
	resul_tuning(generator_path,'DIBCO2010_TL')
 