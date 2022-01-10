import os

os.environ["PYTHONIOENCODING"] = "utf-8"
# 1 geforce
# 0 titan
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from tensorflow.keras import layers
from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import load_model

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
rootPath = 'src/'
pathDibco='datasetsDIBCO/CH_DIBCO/'
pathadd='/home/ubuntu/Sana/Hito-docs/dataset_chunks/'
##########################################################################################################
##########################################################################################################
##########################################################################################################



# define parameters
source = "IAM"
arch = "flor"  ########ne pas modifier, nous utilisons architeture crnn de flor
batch_size = 32
scenario = 'DIBCO_2010'
# define input size, number max of chars per line and list of valid chars
max_text_length = 128  ####not change this value
img_width = 1024  #########for crnn
img_height = 128  #########for crnn
input_size_crnn = (1024, 128, 1)
input_size = (128, 1024, 1)  #############for the GAN
i = 1
flag = 0


##########################################################################################################
##########################################################################################################
##########################################################################################################


def normalizeTranscription(text_line):
    lk = []
    for c in text_line:
        lk.append(c)

    text_line = ' '.join(lk)
    return text_line


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


charset_base = read_file_char(rootPath + 'SetsIAM/CHAR_LIST')
f = codecs.open('charlist.txt', 'w', 'utf-8')
f.writelines(charset_base)
f.close()


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

def unetpp(pretrained_weights=None, input_size=(128, 1024, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #bn = BatchNormalization(momentum=0.8)(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    #bn = BatchNormalization(momentum=0.8)(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    #bn = BatchNormalization(momentum=0.8)(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    #bn = BatchNormalization(momentum=0.8)(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv4)
    drop4 = Dropout(0.5)(bn)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    #bn = BatchNormalization(momentum=0.8)(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv5)
    drop5 = Dropout(0.5)(bn)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    # 	 merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    #bn = BatchNormalization(momentum=0.8)(up6)
    merge6 = concatenate([drop4, bn])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #bn = BatchNormalization(momentum=0.8)(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(bn))
    #bn = BatchNormalization(momentum=0.8)(up7)
    merge7 = concatenate([conv3, bn])
    # 	 merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    #bn = BatchNormalization(momentum=0.8)(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(bn))
    #bn = BatchNormalization(momentum=0.8)(up8)
    merge8 = concatenate([conv2, bn])
    # 	 merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    #bn = BatchNormalization(momentum=0.8)(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(bn))
    # 	 merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    #bn = BatchNormalization(momentum=0.8)(up9)
    merge9 = concatenate([conv1, bn])

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    #bn = BatchNormalization(momentum=0.8)(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn)
    #bn = BatchNormalization(momentum=0.8)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(bn)

    model = Model(inputs=inputs, outputs=conv10)

    # 	 model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def get_optimizer():
    return Adam(lr=1e-4)


def build_crnn():
    ############################# Model Creation########################################
    from network.model import flor

    # create and compile HTRModel
    inputs, outputs = flor(input_size_crnn, len(charset_base) + 1)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    # create and compile
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model


def build_discriminator_1():
	def d_layer(layer_input, filters, f_size=4, bn=True):
		# 		 """Discriminator layer"""
		d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		if bn:
			d = BatchNormalization(momentum=0.8)(d)
			d.trainable=False
		return d

	img_A = Input(shape=(128, 1024, 1))
	img_B = Input(shape=(128, 1024, 1))
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
def readGrayPair(im_name,database_ch):


  deg_image_path = database_ch + '/distorted/' + im_name + '.png'
  original_image = Image.open(deg_image_path)
  original_image = original_image.resize((1024, 128), Image.ANTIALIAS)
  original_image = original_image.convert("RGB")
  grey_image = original_image.convert('L')

  grey_image.save("deg_image.tif")
  deg_image = plt.imread("deg_image.tif")

  gt_image_path = database_ch + '/GT/' + im_name + '.png'
  original_image = Image.open(gt_image_path)
  original_image = original_image.resize((1024, 128), Image.ANTIALIAS)
  original_image = original_image.convert("RGB")
  grey_image = original_image.convert('L')
  grey_image.save("gt_image.tif")
  gt_image = plt.imread("gt_image.tif")
  return deg_image, gt_image

def readGrayPairold(im_name):


  deg_image_path = pathDibco + '/distorted/' + im_name + '.png'
  original_image = Image.open(deg_image_path)
  original_image = original_image.resize((1024, 128), Image.ANTIALIAS)
  original_image = original_image.convert("RGB")
  grey_image = original_image.convert('L')

  grey_image.save("deg_image.tif")
  deg_image = plt.imread("deg_image.tif")

  gt_image_path = pathDibco + '/GT/' + im_name + '.png'
  original_image = Image.open(gt_image_path)
  original_image = original_image.resize((1024, 128), Image.ANTIALIAS)
  original_image = original_image.convert("RGB")
  grey_image = original_image.convert('L')
  grey_image.save("gt_image.tif")
  gt_image = plt.imread("gt_image.tif")
  return deg_image, gt_image

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


###############New GAN######################
def get_gan_network(discriminator_1, generator, optimizer):
    discriminator_1.trainable = False

    gan_input = Input(shape=(128, 1024, 1))  ######### this is the degraded image because it is a cgan

    out_generator = generator(gan_input)
    out_discrimintor_1 = discriminator_1([out_generator, gan_input])  ### remove the gan input 3 from here
    ######################Here we should reshape out_generator to be fed to the RCNN model

    # define composite model
    # out_generator is to compute the BCE loss ....
    # define composite model
    gan = Model([gan_input], [out_discrimintor_1, out_generator])

    gan.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[1, 100],
                optimizer=optimizer)  ##### the weight are to discuss later Please dont forget !!!
    return gan
def list_files_dibco(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append('d ' + name)
    return r
def list_files_dibco_expect2010(dir):
	r = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			if '2010' in name:
				a=100
			elif '2019' in name:
				a=100
			else:
				r.append('d ' + name)
	return r
def list_files_add(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append('o ' +name)
    return r		
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(name)
    return r	
def train_gan(generator, discriminator_1, gan, ep_start=0, epochs=1, batch_size=16):
	# reserve a batch of the training and testing data

	import string
	batch_train = np.zeros((((batch_size, 128, 1024, 1))))
	batch_target = np.zeros((((batch_size, 128, 1024, 1))))

	#list_image_train_dibco = list_files_dibco(pathDibco + '/distorted')
	list_image_train_dibco = list_files_dibco_expect2010(pathDibco + '/distorted')
	list_image_train_dibco=sorted(list_image_train_dibco)
	list_image_train_dibco=list_image_train_dibco[::-1]
	#res1 = list_image_train_dibco[1:12000]
	res1 = list_image_train_dibco[1:12000]
	list_image_train_a = list_files_add(pathadd + '/distorted')
	list_image_train_a=sorted(list_image_train_a)
	list_image_train_a=list_image_train_a[::-1]
	res2 = list_image_train_a[1:3000]
	res=res1+res2
	random.shuffle(res)
	for e in range(ep_start, epochs + 1):
		batch = 0
		print('\n Epoch ', e)
		batch_txt = []

		count_image = 0

		nb = 0
		loss1 = 0
		loss2 = 0
		nbre_batch = 0

		for im in tqdm(res):

			if nb!=-1 :  ###########this conditioning the CRNN recognize
				###################################################which recognize sequence lengh < max_text_length (128)
				dd=im
				ds=dd.split()
				im=ds[1]
				im=im.replace('.png','')
				if(ds[0]=='d'):
					database_ch=pathDibco
				else:
					database_ch=pathadd
				#print(im)
				#print(database_ch)
				deg_image, gt_image = readGrayPair(im,database_ch)
				# print('image found')
				batch_train[batch, :, :, :] = deg_image.reshape(128, 1024, 1)
				batch_target[batch, :, :, :] = gt_image.reshape(128, 1024, 1)

				batch = batch + 1

				if (batch == batch_size):
					# print('Epoch: ', e, ' - Batch: ', nb)
					generated_images = generator.predict(batch_train)
					deg_image1 = batch_train[0].reshape(128,1024)
					gt_image1 = batch_target[0].reshape(128,1024)
					# #here to show current image result 
					prediction1 = generated_images[0].reshape(128,1024)
					plt.imsave("prediction2.png", prediction1, cmap='gray')
					plt.imsave("deg_image1.png", deg_image1, cmap='gray')
					plt.imsave("gt_image1.png", gt_image1, cmap='gray')
					im1=cv2.imread("prediction2.png")
					im2=cv2.imread("deg_image1.png")
					im3=cv2.imread("gt_image1.png")
					show=vconcat_resize([im2,im1,im3])
			
					cv2.imwrite("generationdibco.png", show)
						
						
					valid = np.ones((batch_size,) + (8, 64, 1))
					fake = np.zeros((batch_size,) + (8, 64, 1))
					########### here we train the discriminator
					# print('discriminator_1 training......')
					
 
					discriminator_1.trainable = True
					# '''random add'''
					discriminator_1.train_on_batch([batch_target, batch_train], valid)

					discriminator_1.train_on_batch([generated_images, batch_train], fake)

					discriminator_1.trainable = False
					# print('Training the GAN by freezing the discriminator weights')
					gan.train_on_batch([batch_train], [valid, batch_target])

					nbre_batch = nbre_batch + 1

					batch = 0
					nb = nb + 1

			count_image = count_image + 1
		###################"compute loss per epoch

		print('\n Epoch ', e)

		if (e <= 5 or e % 4 == 0):
			evaluate(e, generator, discriminator_1)
	return generator, discriminator_1,gan


def save(generator, discriminator_1, epoch):

    discriminator_1.save_weights(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/discriminator_weights.h5")
    generator.save_weights(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights/generator_weights.h5")


def evaluate(epoch, generator, discriminator_1):
	list_image_train_dibco = list_files_dibco(pathDibco + '/distorted')
	list_image_train_dibco=sorted(list_image_train_dibco)
	list_image_train_dibco=list_image_train_dibco[::-1]
	res = list_image_train_dibco[0:999]

	count_image = 0
	for im in tqdm(res):
		if count_image >= 0:
			space = np.zeros((128, 1024))

			dd=im
			ds=dd.split()
			im=ds[1]
			im=im.replace('.png','')
			if(ds[0]=='d'):
				database_ch=pathDibco
			else:
				database_ch=pathadd
			#print(im)
			#print(database_ch)
			deg_image, gt_image = readGrayPair(im,database_ch)

			prediction = generator.predict(deg_image.reshape(1, 128, 1024, 1)).reshape(128, 1024)
			plt.imsave("prediction.png", prediction, cmap='gray')
			plt.imsave("deg_image.png", deg_image, cmap='gray')
			plt.imsave("gt_image.png", gt_image, cmap='gray')
			plt.imsave("space.png", space, cmap='gray')
			im1 = cv2.imread("prediction.png")
			im2 = cv2.imread("deg_image.png")
			im3 = cv2.imread("gt_image.png")
			im4 = cv2.imread("space.png")
			show = vconcat_resize([im2, im4, im1, im4, im3])

			if not os.path.exists(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch)):
				os.makedirs(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch))
				os.makedirs(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch) + "/weights")
			cv2.imwrite(rootPath + "/ResultGan" + scenario + "/epoch" + str(epoch) + '/' + im + ".png", show)
	save(generator, discriminator_1, epoch)



def fine_tuning(best_path_for_tuning,nepochs=1,batch_size=8):
	# Freeze all the layers
	print('generator creation..............')
	generator = unet()
	generator.load_weights( best_path_for_tuning + "/weights" + "/generator_weights.h5")

 

	print('discriminator creation..............')
	discriminator_1 = build_discriminator_1()
	discriminator_1.load_weights(best_path_for_tuning + "/weights" + "/discriminator_weights.h5")
	i = 0
	nblayer = len(discriminator_1.layers[:])
 
	
	for layer in discriminator_1.layers[:]:
		if i >= nblayer-2:
		  layer.trainable = True
		else:
		  layer.trainable = False
		i = i + 1
	adam = get_optimizer()
	gan = get_gan_network(discriminator_1, generator, adam)
	generator, discriminator_1, gan = train_gan(generator, discriminator_1, gan,
			      ep_start=0, epochs=nepochs, batch_size=batch_size)



if __name__ == '__main__':

    #############Fine tuning the GAN
    ########set her the path of the best epoch obtained using S2 trained on IAM
    best_path_for_tuning= "ResultGanS2_W10_IAM/epoch144/"
	
    fine_tuning(best_path_for_tuning,nepochs=1,batch_size=8)

