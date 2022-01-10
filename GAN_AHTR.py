import os
os.environ["PYTHONIOENCODING"] = "utf-8"
#1 geforce
#0 titan
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
rootPath='GAN-HTR/'
DatabasePath='/home/ahmed/Desktop/Gan-OCR/Dataset/KHATT/'
scenario='S2_khatt_OP'

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



##########################################################################################################
##########################################################################################################
##########################################################################################################
def get_callbacks(logdir, checkpoint, monitor="loss", verbose=1):
        """Setup the list of callbacks for the model"""

        callbacks = [

            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks

def normalizeTranscription(text_line):
	text_line = text_line.replace('sp', ' sp ')
	text_line = text_line.replace('A', 'A ')
	text_line = text_line.replace('B', 'B ')
	text_line = text_line.replace('E', 'E ')
	text_line = text_line.replace('M', 'M ')
	text_line = text_line.replace('  ', ' ')
	return  text_line

def read_file_shuffle(list_file_path):
	char_file = codecs.open(list_file_path, 'r', 'utf-8')

	list0 = []
	for l in char_file:
		list0.append(l.strip())
	random.shuffle(list0)
	return list0
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
charset_base = read_file_char(rootPath+ 'src/Sets/CHAR_LIST')
f=codecs.open('charlist.txt','w','utf-8')
f.writelines(charset_base)
f.close()


def unet(pretrained_weights=None, input_size=(128,1024, 1)):
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


def build_discriminator_1():

	def d_layer(layer_input, filters, f_size=4, bn=True):
# 		 """Discriminator layer"""
		d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		if bn:
			d = BatchNormalization(momentum=0.8)(d)
		return d

	img_A = Input(shape=(128,1024, 1))
	img_B = Input(shape=(128,1024, 1))
	# img_C = Input(shape=(32,768, 1))
	df = 64
	# Concatenate image and conditioning image by channels to produce input
	combined_imgs = Concatenate(axis=-1)([img_A, img_B])

	d1 = d_layer(combined_imgs, df, bn=False)
	d2 = d_layer(d1, df * 2)
	d3 = d_layer(d2, df * 4)
	d4 = d_layer(d3, df * 4)

	validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

	discriminator = Model([img_A, img_B], validity)
	
	
	discriminator.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
	return discriminator
#######################CRNN CTC Recognize##########################
def ctc_loss_lambda_func(y_true, y_pred):
	"""Function for computing the CTC loss"""

	if len(y_true.shape) > 2:
		y_true = tf.squeeze(y_true)

	# y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
	# output of every model is softmax
	# so sum across alphabet_size_1_hot_encoded give 1
	#               string_length give string length
	input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
	input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

	# y_true strings are padded with 0
	# so sum of non-zero gives number of characters in this string
	label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

	loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

	# average loss across all entries in the batch
	loss = tf.reduce_mean(loss)

	return loss
def build_discriminator_2():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

	 
	return model
 
def build_discriminator_3():


	############################# Model Creation########################################
	from network.model import flor

	# create and compile HTRModel
	inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

	# create and compile
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

	 
	return model	

def readGrayPair(im_name):
	deg_image_path = ('Hito-docs/DatasetKHATT1/' + im_name + '.tif')

	original_image = Image.open(deg_image_path)  # /255.0
	original_image = original_image.resize((1024,128), Image.ANTIALIAS)
	grey_image = original_image.convert('L')
	
	grey_image.save("deg_image2.tif")
	deg_image = plt.imread("deg_image2.tif")
	
	gt_image_path = (DatabasePath +'/Gt/Images/' + im_name + '.tif')
	original_image = Image.open(gt_image_path)  # /255.0
	original_image = original_image.resize((1024,128), Image.ANTIALIAS)
	grey_image = original_image.convert('L')
	grey_image.save("gt_image2.tif")
	gt_image = plt.imread("gt_image2.tif")
	
	return deg_image, gt_image
  
def vconcat_resize(img_list, interpolation  
                   = cv2.INTER_CUBIC): 
      # take minimum width 
    w_min = min(img.shape[1]  
                for img in img_list) 
      
    # resizing images 
    im_list_resize = [cv2.resize(img, 
                      (w_min, int(img.shape[0] * w_min / img.shape[1])), 
                                 interpolation = interpolation) 
                      for img in img_list] 
    # return final image 
    return cv2.vconcat(im_list_resize) 	
 

###############New GAN######################
def get_gan_network(discriminator_1,discriminator_2, generator, optimizer):
	
	discriminator_1.trainable = False
	discriminator_2.trainable = False

	gan_input = Input(shape=(128,1024, 1))  ######### this is the degraded image because it is a cgan

	# input_length = layers.Input(shape=[1], dtype=tf.int32, name='input_length')
	# label_length = layers.Input(name='label_length', shape=[1], dtype=tf.int32)


	out_generator = generator(gan_input)
	out_discrimintor_1 = discriminator_1([out_generator, gan_input])    ### remove the gan input 3 from here 
	######################Here we should reshape out_generator to be fed to the RCNN model
	###################### The RCNN accept shape (1024,128,1)
	reshaped = Reshape((1024,128,1 ), input_shape=(128,1024,1))(out_generator)

	out_discrimintor_2= discriminator_2([reshaped])    ### remove the gan input 3 from here : CRNN Recognizer
	# define composite model
	# out_generator is to compute the BCE loss ....
	# define composite model
	gan = Model([gan_input], [out_discrimintor_1, out_generator,  out_discrimintor_2])

	gan.compile(loss=['mse','binary_crossentropy',ctc_loss_lambda_func], loss_weights=[1,10,1], optimizer=optimizer)   ##### the weight are to discuss later Please dont forget !!!
	return gan



def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		index = charset_base.index(item)
	
		encoded.append(index)
		
	encoded=encoded[::-1]	############this is done only for arabic, otherwise remove this line

	return encoded
def train_gan(generator, discriminator_1,discriminator_2,gan,ep_start=0, epochs=1, batch_size=16):
	 
	# reserve a batch of the training and testing data
	batch_train = np.zeros((((batch_size, 128,1024, 1))))
	batch_target = np.zeros((((batch_size, 128,1024, 1))))
	
	batch_train_gt_path=[]
	
	# Build our GAN netowrks , 2 gans
	#fc=codecs.open('histo.txt','w+','utf-8')
	list_image_train = read_file_shuffle(rootPath + 'src/Sets/list_train')
	#res = list_image_train[-16:] 
	res = list_image_train
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	for e in range(ep_start, epochs + 1):
		batch = 0
		print ('\n Epoch ', e)
		batch_txt = []
		divider = random.randint(300, 600)
		count_image=0


		nb=0
		loss1=0
		loss2=0
		nbre_batch=0
		
		for im in tqdm(res):
			###########read Grund truth text
			matched_lines = [s for s in list_lines if im in s]
			#print(matched_lines)
			l = matched_lines[0]
			l1 = l.split()
			text_line = l1[8]
			line = normalizeTranscription(text_line)
			len_trancription=len(line.split())
			if len_trancription < max_text_length : ###########this conditioning the CRNN recognize 
			###################################################which recognize sequence lengh < max_text_length (128)

				batch_txt.append(line)
				
				########## read image pixels
				deg_image, gt_image = readGrayPair(im)
				#print('image found')
				batch_train[batch, :, :, :] = deg_image.reshape(128,1024, 1)
				batch_target[batch, :, :, :] = gt_image.reshape(128,1024, 1)
				
				batch = batch + 1
				
				batch_train_gt_path.append(DatabasePath +'/Gt/Images/' + im + '.tif')
				if (batch == batch_size):
						#print('Epoch: ', e, ' - Batch: ', nb)
						generated_images = generator.predict(batch_train)
						#deg_image1 = batch_train[0].reshape(128,1024)
						#gt_image1 = batch_target[0].reshape(128,1024)
						#here to show current image result 
						#prediction1 = generated_images[0].reshape(128,1024)
						#plt.imsave("prediction2.png", prediction1, cmap='gray')
						#plt.imsave("deg_image1.png", deg_image1, cmap='gray')
						#plt.imsave("gt_image1.png", gt_image1, cmap='gray')
						#im1=cv2.imread("prediction2.png")
						#im2=cv2.imread("deg_image1.png")
						#im3=cv2.imread("gt_image1.png")
						#show=vconcat_resize([im2,im1,im3])
				
						#cv2.imwrite("generation.png", show)
						################## prepare discriminator labels
						
				
						valid = np.ones((batch_size,) + ( 8, 64, 1))
						fake = np.zeros((batch_size,) + ( 8, 64, 1))
						########### here we train the discriminator 
						# print('discriminator_1 training......')	
						discriminator_1.trainable = True
						# '''random add'''
						d1=discriminator_1.train_on_batch([batch_target, batch_train], valid)
						#f1.write('epoch ' + str(e) + ' batch ' + str(nb) + ' loss ' + str(d1) + '\n')
						d2=discriminator_1.train_on_batch([generated_images, batch_train], fake)
						#f2.write('epoch ' + str(e) + ' batch ' + str(nb) + ' loss ' + str(d2)+ '\n')
						###### here train your rcnn (discriminator_2) with a real batch (GT images)
						discriminator_2.trainable = True
						#################preapare data for the crnn

						x_train_rcnn=[] ############images in batch
						y_train_rcnn=[] ##ground truth of this batch
						#fc.write( 'Epoch: ' +  str(e) +  ' - Batch: ' + str(nb) + '\n')
						for i in range (batch_size):
							#print(batch_train_gt_path[i])
							img=pp.preprocess(batch_train_gt_path[i],input_size_crnn)
							x_train_rcnn.append(img)
							#print(batch_txt[i])
							#fc.write(batch_train_gt_path[i] + '\n')
							#fc.write(batch_txt[i] + '\n')
							encoded_txt=encode_txt(batch_txt[i])
							y_train_rcnn.append(encoded_txt)
							del img
							del encoded_txt
							
						y_train_rcnn = [np.pad(y, (0, max_text_length - len(y))) for y in y_train_rcnn]
						y_train_rcnn = np.asarray(y_train_rcnn, dtype=np.int16)
						x_train_rcnn=pp.normalization(x_train_rcnn)
						#### data to be fed to the CRNN network for training

						
						############################## Training recognizer RCNN ####################################################################

			 
						callbacks1 = get_callbacks(logdir=output_path, checkpoint=target_path, verbose=0)
						d3=discriminator_2.fit(x_train_rcnn,y_train_rcnn,batch_size=batch_size,initial_epoch=e, epochs=e +1, verbose=0,
                             callbacks=callbacks1,shuffle=True)
						#f3.write('epoch ' + str(e) + ' batch ' + str(nb) + ' loss ' + str(d3.history['loss'])+ '\n')

					
						# print('End discriminator_2 training.')	
						############################## End of Training recognizer RCNN ####################################################################
						########### train the generator with GAN  , by freezing the discriminator weights  
						discriminator_2.trainable = False
						discriminator_1.trainable = False
						# print('Training the GAN by freezing the discriminator weights')	
						g_loss=gan.train_on_batch([batch_train], [valid, batch_target,y_train_rcnn])
						#fg.write('epoch ' + str(e) + ' batch ' + str(nb) + ' loss ' + str(g_loss)+ '\n')
						del y_train_rcnn
						del x_train_rcnn
						
						##############################################################################################################
						##############################################################################################################
						# ###### here train your rcnn progressively for recognition purpose
						# discriminator_3.trainable = True
						# #################preapare data for the crnn

						# x_train_rcnn_p=[] ############images in batch
						# y_train_rcnn_p=[] ##ground truth of this batch

						# for j in range (batch_size):
							# #print(generated_images[i])
							# pred = generated_images[j].reshape(128,1024)
							# plt.imsave("pred1.png", pred, cmap='gray')
							# n=batch_train_gt_path[j]
							# g=cv2.imread(n)
							# height, width,c = g.shape
							# predx = Image.open("pred1.png") 
							# predx = predx.convert('L')
							# predx = predx.resize((width,height), Image.ANTIALIAS)
							# predx.save("pred.tif")
							 
							# ########"ici les images pour apprendre le crnn sont genereted via generator ###################################
							# imgp=pp.preprocess("pred.tif",input_size_crnn)
							# #print(imgp)
							# x_train_rcnn_p.append(imgp)
							# del imgp
							# encoded_txt=encode_txt(batch_txt[j])
							# #print(encoded_txt)
							# y_train_rcnn_p.append(encoded_txt)
							# del encoded_txt
						# y_train_rcnn_p = [np.pad(y, (0, max_text_length - len(y))) for y in y_train_rcnn_p]
						# y_train_rcnn_p = np.asarray(y_train_rcnn_p, dtype=np.int16)

						# x_train_rcnn_p=pp.normalization(x_train_rcnn_p)
						# #### data to be fed to the CRNN network for training
						 
						
						# #d4=discriminator_3.train_on_batch(x_train_rcnn_p,y_train_rcnn_p)
						# callbacks2 = get_callbacks(logdir=output_path2, checkpoint=target_path2, verbose=0)
						# d4=discriminator_3.fit(x_train_rcnn_p,y_train_rcnn_p,batch_size=batch_size,initial_epoch=e, epochs=e +1, verbose=0,
                             # callbacks=callbacks2,shuffle=True)						
						# f4.write('epoch ' + str(e) + ' batch ' + str(nb) + ' loss ' + str(d4.history['loss'])+ '\n')
	
						
						# del y_train_rcnn_p
						# del x_train_rcnn_p

						###########################end crnn training#################################################################
						##############################################################################################################
						##############################################################################################################
						
						nbre_batch=nbre_batch+1
						batch_train_gt_path=[]
						batch_txt=[]
						batch = 0
						nb=nb+1

			count_image=count_image+1
		###################"compute loss per epoch

		print ('\n Epoch ', e)
		
		if (e <= 5 or e % 4 == 0):
			evaluate(e, generator, discriminator_1,discriminator_2,gan)
	return generator, discriminator_1,discriminator_2,gan


def save(gan, generator, discriminator_1,discriminator_2,epoch):

	

	
	gan.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/gan_weights.h5")	
	
	discriminator_1.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/discriminator_weights.h5")
	discriminator_2.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn_weights.h5")
	#discriminator_3.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn_progressive_weights.h5")
	generator.save_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")

def load(epoch):
	generator = unet()
	generator = generator.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")
	discriminator_1 = build_discriminator_1()
	discriminator_1.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/discriminator_weights.h5")
	 
	discriminator_2 = build_discriminator_2()
	discriminator_2.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/rcnn_weights.h5")
	 
	adam = get_optimizer()
	gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	
	#gan = gan.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/gan_weights.h5")
	return gan, generator, discriminator_1,discriminator_2,discriminator_3
	
def evaluate(epoch, generator, discriminator_1,discriminator_2,gan):
	
	list_image_valid = read_file(rootPath + 'src/Sets/list_valid')
	#res = list_image_valid[-2:] 
	res = list_image_valid
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	count_image=0
	for im in res:
		if count_image >=0:
			space = np.zeros((128,1024))
			deg_image, gt_image = readGrayPair(im)

			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction.png", prediction, cmap='gray')
			plt.imsave("deg_image.png", deg_image, cmap='gray')
			plt.imsave("gt_image.png", gt_image, cmap='gray')
			plt.imsave("space.png", space, cmap='gray')
			im1=cv2.imread("prediction.png")
			im2=cv2.imread("deg_image.png")
			im3=cv2.imread("gt_image.png")
			im4=cv2.imread("space.png")
			show = vconcat_resize([im2, im4, im1, im4, im3])
		
			if not os.path.exists(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch)):
				os.makedirs(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch))
				os.makedirs(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights")
			cv2.imwrite(rootPath+"/ResultGan" + scenario + "/epoch" + str(epoch) + '/'+  im + ".png", show)
	save(gan, generator, discriminator_1,discriminator_2,epoch)

def train_GAN_crnn(nepochs,batch_size):
	print('generator creation..............')
	generator = unet()
	print('discriminator 1 creation..............')
	discriminator_1 = build_discriminator_1()
	###########"load data for RCNN
	print('discriminator 2 creation..............')
	####discriminator_2 : cest le crnn
	discriminator_2 = build_discriminator_2()
	print('discriminator 3 creation..............')	
	discriminator_3 = build_discriminator_3()
	epo = 0
	adam = get_optimizer()
	gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	generator, discriminator_1,discriminator_2,gan = train_gan(generator, discriminator_1,discriminator_2,gan, ep_start=0, epochs=nepochs, batch_size=batch_size)
def resume_train_GAN_crnn(nepochs,epo,batch_size):


	generator = unet()
	generator.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/generator_weights.h5")
	discriminator_1 = build_discriminator_1()
	discriminator_1.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/discriminator_weights.h5")
	 
	discriminator_2 = build_discriminator_2()
	discriminator_2.load_weights(rootPath+"/ResultGan" + scenario + "/epoch" + str(epo-1) + "/weights/rcnn_weights.h5")
	 
	adam = get_optimizer()
	gan = get_gan_network(discriminator_1,discriminator_2, generator, adam)
	generator, discriminator_1,discriminator_2,gan = train_gan(generator, discriminator_1,discriminator_2, gan,ep_start=epo, epochs=nepochs, batch_size=batch_size)


def loadCRNNModel(epoch,mode_crnn='no_progressive'):
	from data.generator import DataGenerator
	input_size = (1024, 128, 1)
	dtgen = DataGenerator(source=source_path,
						  batch_size=batch_size,
						  charset=charset_base,
						  max_text_length=max_text_length)


	from network.model import HTRModel

	# create and compile HTRModel
	model = HTRModel(architecture=arch,
					 input_size=input_size,
					 vocab_size=dtgen.tokenizer.vocab_size,
					 beam_width=10,
					 stop_tolerance=20,
					 reduce_tolerance=15)

	model.compile(learning_rate=0.001)
	model.summary(output_path, "summary.txt")

	# get default callbacks and load checkpoint weights file (HDF5) if exists
	if mode_crnn=='progressive':
		model.load_checkpoint(target='handwritten-text-recognition/ResultGanS3_khatt_OP/epoch128/weights/rcnn_weights.h5')
	else:
		model.load_checkpoint(target='handwritten-text-recognition/output-KHATT-GT/khatt/flor/checkpoint_weights.hdf5')
	return dtgen,model
def ocr_crnn(filename,dtgen,model):
	text = ''
	input_size = (1024, 128, 1)

	im=pp.preprocess(filename,input_size)
	x_test = []
	x_test.append(im)
	x_test=pp.normalization(x_test)

	# predict() function will return the predicts with the probabilities
	predicts, _ = model.predict(x=x_test,
								use_multiprocessing=False,
								ctc_decode=True,
								verbose=0)

	# decode to string
	predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
	text=predicts[0]
	s=text.split()
	s=s[::-1]
	reco=' '.join(s)
	reco=reco.strip()
	print(reco)
	return reco
def predict_gan(epoch, generator,list_image_valid,set):
	
	

	count_image=0
	for im in list_image_valid:
		if count_image >=0:

			#deg_image, gt_image = readGrayPairPad(im)
			original_path_image_gt=DatabasePath + '/Gt/Images/' + im + '.tif'
			claen_image=cv2.imread(original_path_image_gt)
			noisy_image_path='Hito-docs/DatasetKHATT1/' + im + '.tif'
			noisy_image=cv2.imread(noisy_image_path)
			
			#height, width,c = noisy_image.shape
			#############resize the height of noisy image to 32
			############add padding

			#noisy_image=addpad_image(noisy_image)
			height, width,c = noisy_image.shape
			#cv2.imwrite('out_padded.tif',noisy_image)
			##############end padding
			
			
			
			
			
			grey_image = original_image.convert('L')
			grey_image.save("deg_image3.tif")
			deg_image = plt.imread("deg_image3.tif")
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction3.png", prediction, cmap='gray')
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch)):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch))
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/prediction")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/prediction_reduced")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/visualize")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Truth")
			################"resize predicted image to original size
			cv2.imwrite(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/Truth/'+  im + ".tif",claen_image)
			original_image = Image.open('prediction3.png') 
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/prediction_reduced/'+  im + ".tif")
			########################""resizingggggggggg	
			original_image = original_image.resize((width,height), Image.ANTIALIAS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + '/prediction/'+  im + ".tif")
			# ######################space image
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted"):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted")
			
			
			original_image = Image.open(noisy_image_path) 
			original_image = original_image.resize((1024,128), Image.ANTIALIAS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/set_" + set + "_epoch_" + str(epoch) + "/Distorted/" + im + ".tif")
			
			 
		count_image=count_image+1
def predict_gan_hard(epoch, generator,list_image_valid,set):
	
	
	scenario='S2_khatt_OP'

	count_image=0
	for im in list_image_valid:
		if count_image >=0:

			#deg_image, gt_image = readGrayPairPad(im)
			original_path_image_gt=DatabasePath + '/Gt/Images/' + im + '.tif'
			claen_image=cv2.imread(original_path_image_gt)
			noisy_image_path='Hito-docs/DatasetKHATT1_hard3/' + im + '.tif'
			noisy_image=cv2.imread(noisy_image_path)
			
			#height, width,c = noisy_image.shape
			#############resize the height of noisy image to 32
			############add padding

			#noisy_image=addpad_image(noisy_image)
			height, width,c = noisy_image.shape
			#cv2.imwrite('out_padded.tif',noisy_image)
			##############end padding
			
			
			
			
			
			original_image = Image.open(noisy_image_path) 
			original_image = original_image.resize((1024,128), Image.ANTIALIAS)

			
			grey_image = original_image.convert('L')
			grey_image.save("deg_image3x.tif")
			deg_image = plt.imread("deg_image3x.tif")
			
			prediction = generator.predict(deg_image.reshape(1, 128,1024, 1)).reshape(128,1024)
			plt.imsave("prediction3x.png", prediction, cmap='gray')
			if not os.path.exists(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch)):
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch))
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/prediction")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/prediction_reduced")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/visualize")
				os.makedirs(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + "/Truth")
			################"resize predicted image to original size
			#cv2.imwrite(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/Truth/'+  im + ".tif",claen_image)
			original_image = Image.open('prediction3x.png') 
			#original_image.save(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/prediction_reduced/'+  im + ".tif")
			########################""resizingggggggggg	
			original_image = original_image.resize((width,height), Image.ANTIALIAS)
			original_image.save(rootPath+ "/ResultGan" + scenario + "/hard3_set_" + set + "_epoch_" + str(epoch) + '/prediction/'+  im + ".tif")
			######################space image
			
			 
		count_image=count_image+1
def addpad_image(img):

	# convert each image of shape (32, 128, 1)
	w, h,c = img.shape
	#print(h)
	white =   [255,255,255]
	
	w_ad=1024-h
	if h < 1024:

		return cv2.copyMakeBorder(img,0,0,w_ad,0,cv2.BORDER_CONSTANT,value=white)
	else:
		return img	

		
def recognition_hard(list, set,epoch,mode_crnn):
	if mode_crnn=='progressive':
		path_test=rootPath + '/ResultGan' + scenario + '/hard3_set_' + set + '_epoch_' + str(epoch)+ '/prediction/'
	else:
		path_test=rootPath + '/ResultGan' + scenario + '/hard3_set_' + set + '_epoch_' + str(epoch)+ '/prediction/'
	
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	dtgen,model=loadCRNNModel(epoch,mode_crnn)
	list_image_valid = read_file(rootPath + 'src/Sets/' + list)
	list_reco_c=[]
	list_reco_w=[]
	list_truth_c=[]
	list_truth_w=[]	
	
	for im in list_image_valid:
		matched_lines = [s for s in list_lines if im in s]
		#print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		text_line=normalizeTranscription(text_line)
		truth_char=text_line
		li=text_line.split()
		print(len(li))
		if len(li) < 128:
			gen_txt = ocr_crnn( path_test + im + '.tif',dtgen,model)
			list_reco_c.append(gen_txt + '\n')
			list_truth_c.append(truth_char+ '\n')
			words=gen_txt
			words=words.replace(' ' ,'')
			words=words.replace(' ' ,'')
			words=words.replace('sp' ,' ')
			print(words)
			list_reco_w.append(words+ '\n')
			twords=truth_char
			twords=twords.replace(' ' ,'')
			twords=twords.replace(' ' ,'')
			twords=twords.replace('sp' ,' ')
			list_truth_w.append(twords+ '\n')		

			
			
	path_result=path_test.replace('prediction','results_' + mode_crnn)
	if not os.path.exists(path_result):
		os.makedirs(path_result)
	
	f=codecs.open(path_result + 'c_reco_' + set + '.txt','w','utf-8')
	f.writelines(list_reco_c)
	f.close()

	f=codecs.open(path_result + 'c_truth_' + set + '.txt','w','utf-8')
	f.writelines(list_truth_c)
	f.close()

	f1=codecs.open(path_result + 'w_reco_' + set + '.txt','w','utf-8')
	f1.writelines(list_reco_w)
	f1.close()	
	
	f1=codecs.open(path_result + 'w_truth_' + set + '.txt','w','utf-8')
	f1.writelines(list_truth_w)
	f1.close()		
	#####################compute result CER%
	command3 = 'wer -a -e  ' + path_result + 'c_truth_' + set + '.txt' + ' ' + path_result + 'c_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_CER.txt'
	os.system(command3)	
	#####################compute result WER%
	command2 = 'wer -a -e ' + path_result + 'w_truth_' + set + '.txt' + ' ' + path_result + 'w_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_WER.txt'
	os.system(command2)			
def recognition(list, set,epoch,mode_crnn):
	if mode_crnn=='progressive':
		path_test=rootPath + '/ResultGan' + scenario + '/set_' + set + '_epoch_' + str(epoch)+ '/prediction/'
	else:
		path_test=rootPath + '/ResultGan' + scenario + '/set_' + set + '_epoch_' + str(epoch)+ '/prediction/'
	
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	dtgen,model=loadCRNNModel(epoch,mode_crnn)
	list_image_valid = read_file(rootPath + 'src/Sets/' + list)
	list_reco_c=[]
	list_reco_w=[]
	list_truth_c=[]
	list_truth_w=[]	
	
	for im in list_image_valid:
		matched_lines = [s for s in list_lines if im in s]
		#print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		text_line=normalizeTranscription(text_line)
		truth_char=text_line
		li=text_line.split()
		print(len(li))
		if len(li) < 128:
			gen_txt = ocr_crnn( path_test + im + '.tif',dtgen,model)
			list_reco_c.append(gen_txt + '\n')
			list_truth_c.append(truth_char+ '\n')
			words=gen_txt
			words=words.replace(' ' ,'')
			words=words.replace(' ' ,'')
			words=words.replace('sp' ,' ')
			print(words)
			list_reco_w.append(words+ '\n')
			twords=truth_char
			twords=twords.replace(' ' ,'')
			twords=twords.replace(' ' ,'')
			twords=twords.replace('sp' ,' ')
			list_truth_w.append(twords+ '\n')		

			
			
	path_result=path_test.replace('prediction','results_' + mode_crnn)
	if not os.path.exists(path_result):
		os.makedirs(path_result)
	
	f=codecs.open(path_result + 'c_reco_' + set + '.txt','w','utf-8')
	f.writelines(list_reco_c)
	f.close()

	f=codecs.open(path_result + 'c_truth_' + set + '.txt','w','utf-8')
	f.writelines(list_truth_c)
	f.close()

	f1=codecs.open(path_result + 'w_reco_' + set + '.txt','w','utf-8')
	f1.writelines(list_reco_w)
	f1.close()	
	
	f1=codecs.open(path_result + 'w_truth_' + set + '.txt','w','utf-8')
	f1.writelines(list_truth_w)
	f1.close()		
	#####################compute result CER%
	command3 = 'wer -a -e  ' + path_result + 'c_truth_' + set + '.txt' + ' ' + path_result + 'c_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_CER.txt'
	os.system(command3)	
	#####################compute result WER%
	command2 = 'wer -a -e ' + path_result + 'w_truth_' + set + '.txt' + ' ' + path_result + 'w_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_WER.txt'
	os.system(command2)		
def evaluateTest(epoch,list,set):
	list_image_valid = read_file(rootPath + 'src/Sets/' + list)
	print('generator creation..............')
	generator = unet()
	generator.load_weights(rootPath+ "/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")

	predict_gan(epoch, generator,list_image_valid,set)	
def evaluateTest_hard(epoch,list,set):
	list_image_valid = read_file(rootPath + 'src/Sets/' + list)
	print('generator creation..............')
	generator = unet()
	scenario='S2_khatt_OP'

	generator.load_weights(rootPath+ "/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")

	predict_gan_hard(epoch, generator,list_image_valid,set)	
def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)

def computeloss(d_loss_all,index=5):
	epochs=[]
	loss_val=[]
	for line_l in d_loss_all:
		line=line_l.replace('[','')
		line=line_l.replace(']','')
		line=line_l.replace(',','')
		s=line.split()
		epochs.append(s[1])
		loss_val.append(s[1] + '_ ' + s[index])
	ep=unique(epochs)
	l=[]
	for e in ep:
		m=e + '_'
		matched_lines = [s for s in loss_val if m in s]
		accum=0
		co=0
		for mat in matched_lines:
			val_s=mat.split('_ ')
			val=val_s[1]
			val=val.replace('[','')
			val=val.replace(']','')
			fval=float(val)
			accum=accum + fval
			co=co+1
		moy=accum/co
		l.append(moy)
	return l	
		
	
def plot_curves():
	g_loss=[]
	d_loss_real=[]
	d_loss_fake=[]
	crnn_loss=[]
	crnn_progressive_loss=[]
	############loop over file txt and compute losses per epoch
	d_loss_real_all=read_file('d_loss_realS3.txt')
	d_loss_real=computeloss(d_loss_real_all,index=5)
	
	d_loss_fake_all=read_file('d_loss_fakeS3.txt')
	d_loss_fake=computeloss(d_loss_fake_all,index=5)
	
	
	
	g_loss_all=read_file('gan_lossS3.txt')
	g_loss=computeloss(g_loss_all,index=6)
	
	

 
	
	
	############################"end computing###########
	import pandas as pd
	####################figure des loss de genartor and discriminator
	ax = pd.DataFrame(
		{
			'Generative Loss': g_loss,
			'Discriminative Loss Real': d_loss_real,

		}
	).plot(title='Training loss', logy=True)
	ax.set_xlabel("Epochs")
	ax.set_ylabel("Loss")
	# the plot gets saved to 'output.png'
	fig1=ax.get_figure()
	fig1.savefig('train_losses_gan_S2_opt.png')
	
	
	
	crnn_loss_all=read_file('crnn_lossS3.txt')
	
	
	crnn_loss=computeloss(crnn_loss_all,index=5)
	
	crnn_progressive_loss_all=read_file('crnn_pg_lossS3.txt')
	crnn_progressive_loss=computeloss(crnn_progressive_loss_all,index=5)
	####################figure des loss de CRNN and CRNN progressive
	axc = pd.DataFrame(
		{

			'CNN-GRU Loss': crnn_loss,
			'CNN-GRU Loss Progressive': crnn_progressive_loss,
		}
	).plot(title='Training loss', logy=True)
	axc.set_xlabel("Epochs")
	axc.set_ylabel("Loss")
	fig1=axc.get_figure()
	# the plot gets saved to 'output.png'
	fig1.savefig('train_losses_crnn_S2_opt.png')	
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r	
def resizetruth():
	
	p='handwritten-text-recognition/ResultGanS2_W1/set_test_epoch_104/Truth/'
	ps='handwritten-text-recognition/ResultGanS2_W1/set_test_epoch_104/Truth_reduced/'
	l=list_files(p)
	
	for im in l:
		original_image = Image.open(im)
		##get name
		head, tail = os.path.split(im)
		original_image = original_image.resize((1024,128), Image.ANTIALIAS)
		grey_image = original_image.convert('L')
		grey_image.save(ps + tail)	


	p='handwritten-text-recognition/ResultGanS2_W1/set_dev_epoch_104/Truth/'
	ps='handwritten-text-recognition/ResultGanS2_W1/set_dev_epoch_104/Truth_reduced/'
	l=list_files(p)
	
	for im in l:
		original_image = Image.open(im)
		##get name
		head, tail = os.path.split(im)
		original_image = original_image.resize((1024,128), Image.ANTIALIAS)
		grey_image = original_image.convert('L')
		grey_image.save(ps + tail)		
		
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def get_psnr_khatt():
	DatabasePathGT = 'handwritten-text-recognition/ResultsSauvegarde/ResultGanS2_W0p5/set_test_epoch_92/Truth/'

	count_image = 1
	qo = 0
	recap = 0
	list_image= read_file('handwritten-text-recognition/src/Sets/list_test')
	for im in list_image:
		original_path_image_gt = DatabasePathGT + '/' + im + '.tif'
		original_image = Image.open(original_path_image_gt)
		original_image = original_image.resize((1024,128), Image.ANTIALIAS)
		grey_image = original_image.convert('1')
		grey_image.save("gt.png")
		gt = plt.imread("gt.png")

		enhanced_image_path = 'handwritten-text-recognition/ResultGanS2_khatt_OP/hard3_set_test_epoch_112/prediction/'+  im + ".tif"
		im2 = Image.open(enhanced_image_path)
		im2 = im2.resize((1024,128), Image.ANTIALIAS)
		im2 = im2.convert('1')
		im2.save('im2.png')
		predicted = plt.imread('im2.png')

		psnrv = psnr(predicted, gt)
		print(psnrv)
		recap = recap + psnrv
		qo += 1
	av = recap / qo
	print('average psnr: ')
	print(av)				
if __name__ == '__main__':
 
	#############Train the GAN
	train_GAN_crnn(150,8)
	
 