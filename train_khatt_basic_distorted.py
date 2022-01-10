import os
os.environ["PYTHONIOENCODING"] = "utf-8"
#1 geforce
#0 titan
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

from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.models import load_model

import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.compat.v1.Session(config=config)


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
#from computeMetric import computeMetrics
import string
################################modifiier ici le path des dossiers##################################################
##########################################################################################################
##########################################################################################################
rootPath='handwritten-text-recognition/'
DatabasePath='BaseKHATT/DatasetKHATT/'
##########################################################################################################
##########################################################################################################
##########################################################################################################





# define parameters
source = "khatt"
arch = "flor" ########ne pas modifier, nous utilisons architeture crnn de flor
batch_size=32
# define paths
source_path = os.path.join("..", "data", f"{source}.hdf5")
output_path = os.path.join("..", "output-KHATT-distorted"   , source, arch)
target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
os.makedirs(output_path, exist_ok=True)





# define input size, number max of chars per line and list of valid chars 
max_text_length = 128  ####not change this value
img_width=1024 #########for crnn
img_height=128 #########for crnn
input_size_crnn = (1024,128, 1)
input_size = (128,1024, 1) #############for the GAN




##########################################################################################################
##########################################################################################################
##########################################################################################################
def get_callbacks(logdir, checkpoint, monitor="val_loss", verbose=1):
        """Setup the list of callbacks for the model"""

        callbacks = [

          CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=15,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=20,
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

####################################
#charlatin =str(string.printable[:95])
#f=codecs.open(rootPath+ 'src/SetsKHATT/CHAR_LIST','w','utf-8')
#for s in charlatin:
	#print(s)
	#f.write(s +'\n')
#f.close()
####################################
charset_base=read_file_char(rootPath+ 'src/Sets/CHAR_LIST')
#print(charset_base)

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


def encode_txt(text):
	encoded=[]
	cc=text.split()
	for item in cc:
		index = charset_base.index(item)
	
		encoded.append(index)
		
	encoded=encoded[::-1]	############this is done only for arabic, otherwise remove this line

	return encoded

def train_crnn( crnn ,ep_start=0, epochs=130, batch_size=32):



	batch_train_gt_path=[]

	list_image_train = read_file_shuffle(rootPath + 'src/Sets/list_train')
	list_image_valid = read_file_shuffle(rootPath + 'src/Sets/list_valid')
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')

	batch_txt = []
	nb=0
	x_train_rcnn=[] ############images in batch
	y_train_rcnn=[] ##ground truth of this batch
	for im in tqdm(list_image_train):
		###########read Grund truth text
		matched_lines = [s for s in list_lines if im in s]
		#print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		line = normalizeTranscription(text_line)
		len_trancription=len(line.split())
		if len_trancription < max_text_length : ###########this conditioning the CRNN recognize 

			batch_train_gt_path = []
			batch_train_gt_path.append('Hito-docs/DatasetKHATT1/' + im + '.tif')
			#print(batch_train_gt_path)
			imgx=pp.preprocess(batch_train_gt_path[0],input_size_crnn)
			x_train_rcnn.append(imgx)
			encoded_txt=encode_txt(line)
			y_train_rcnn.append(encoded_txt)
			del imgx
			del encoded_txt


			
			batch_train_gt_path=[]
			batch_txt=[]
			batch = 0
		nb=nb+1
	y_train_rcnn = [np.pad(y, (0, max_text_length - len(y))) for y in y_train_rcnn]
	y_train_rcnn = np.asarray(y_train_rcnn, dtype=np.int16)
	x_train_rcnn = np.asarray(x_train_rcnn)
	x_train_rcnn = pp.normalization(x_train_rcnn)

	#################################################################################################"
	batch_txt = []
	nb=0
	x_valid_rcnn=[] ############images in batch
	y_valid_rcnn=[] ##ground truth of this batch
	batch_valid_gt_path=[]
	for im in tqdm(list_image_valid):
		###########read Grund truth text
		matched_lines = [s for s in list_lines if im in s]
		#print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		line = normalizeTranscription(text_line)
		len_trancription=len(line.split())
		if len_trancription < max_text_length : ###########this conditioning the CRNN recognize 

			batch_valid_gt_path = []
			batch_valid_gt_path.append('Hito-docs/DatasetKHATT1/' + im + '.tif')
			img=pp.preprocess(batch_valid_gt_path[0],input_size_crnn)
			x_valid_rcnn.append(img)
			encoded_txt=encode_txt(line)
			y_valid_rcnn.append(encoded_txt)
			del img
			del encoded_txt


			batch_valid_gt_path=[]
			batch_txt=[]
			batch = 0
		nb=nb+1
	y_valid_rcnn = [np.pad(y, (0, max_text_length - len(y))) for y in y_valid_rcnn]
	y_valid_rcnn = np.asarray(y_valid_rcnn, dtype=np.int16)
	x_valid_rcnn = pp.normalization(x_valid_rcnn)



	validation_data=(x_valid_rcnn,y_valid_rcnn)


	callbacks = get_callbacks(logdir=output_path, checkpoint=target_path, verbose=0)
	crnn.fit(x_train_rcnn,y_train_rcnn, validation_data=validation_data,batch_size=batch_size,initial_epoch=ep_start, epochs=epochs, verbose=1,
		 callbacks=callbacks,shuffle=True,validation_freq=1)
	return   crnn



 
def train(nepochs,batch_size):
	crnn = build_crnn() 
	crnn = train_crnn(crnn, ep_start=0, epochs=nepochs, batch_size=batch_size)
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

def loadCRNNModel():
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

	model.load_checkpoint(target='handwritten-text-recognition/output-KHATT-distorted/khatt/flor/checkpoint_weights.hdf5')
	return dtgen,model
def recognition_hard():
	path_test = 'Hito-docs/DatasetKHATT1_hard3/'
	set='distorted_hard3'
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	dtgen, model = loadCRNNModel()
	list_image_valid = read_file(rootPath + 'src/Sets/list_test')
	list_reco_c = []
	list_reco_w = []
	list_truth_c = []
	list_truth_w = []
	i=0
	for im in list_image_valid:
		matched_lines = [s for s in list_lines if im in s]
		# print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		text_line = normalizeTranscription(text_line)
		truth_char = text_line
		li = text_line.split()
		
		if len(li) < 128:
			print('im : ' + str(i))
			i=i+1
			gen_txt = ocr_crnn(path_test + im + '.tif', dtgen, model)
			list_reco_c.append(gen_txt + '\n')
			list_truth_c.append(truth_char + '\n')
			words = gen_txt
			words = words.replace(' ', '')
			words = words.replace(' ', '')
			words = words.replace('sp', ' ')
			
			list_reco_w.append(words + '\n')
			twords = truth_char
			twords = twords.replace(' ', '')
			twords = twords.replace(' ', '')
			twords = twords.replace('sp', ' ')
			list_truth_w.append(twords + '\n')

	path_result ='handwritten-text-recognition/output-KHATT-distorted/khatt/flor/hard3'
	if not os.path.exists(path_result):
		os.makedirs(path_result)

	f = codecs.open(path_result + 'c_reco_' + set + '.txt', 'w', 'utf-8')
	f.writelines(list_reco_c)
	f.close()

	f = codecs.open(path_result + 'c_truth_' + set + '.txt', 'w', 'utf-8')
	f.writelines(list_truth_c)
	f.close()

	f1 = codecs.open(path_result + 'w_reco_' + set + '.txt', 'w', 'utf-8')
	f1.writelines(list_reco_w)
	f1.close()

	f1 = codecs.open(path_result + 'w_truth_' + set + '.txt', 'w', 'utf-8')
	f1.writelines(list_truth_w)
	f1.close()
	#####################compute result CER%
	command3 = 'wer -a -e  ' + path_result + 'c_truth_' + set + '.txt' + ' ' + path_result + 'c_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_CER.txt'
	os.system(command3)
	#####################compute result WER%
	command2 = 'wer -a -e ' + path_result + 'w_truth_' + set + '.txt' + ' ' + path_result + 'w_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_WER.txt'
	os.system(command2)

def recognition_easy():
	path_test = 'Hito-docs/DatasetKHATT1/'
	
	
	
	set='distorted_default_'
	list_lines = read_file(rootPath + 'src/Sets/lines.txt')
	dtgen, model = loadCRNNModel()
	list_image_valid = read_file(rootPath + 'src/Sets/list_test')
	list_reco_c = []
	list_reco_w = []
	list_truth_c = []
	list_truth_w = []
	i=0
	for im in list_image_valid:
		matched_lines = [s for s in list_lines if im in s]
		# print(matched_lines)
		l = matched_lines[0]
		l1 = l.split()
		text_line = l1[8]
		text_line = normalizeTranscription(text_line)
		truth_char = text_line
		li = text_line.split()
		
		if len(li) < 128:
			print('im : ' + str(i))
			i=i+1
			gen_txt = ocr_crnn(path_test + im + '.tif', dtgen, model)
			list_reco_c.append(gen_txt + '\n')
			list_truth_c.append(truth_char + '\n')
			words = gen_txt
			words = words.replace(' ', '')
			words = words.replace(' ', '')
			words = words.replace('sp', ' ')
			
			list_reco_w.append(words + '\n')
			twords = truth_char
			twords = twords.replace(' ', '')
			twords = twords.replace(' ', '')
			twords = twords.replace('sp', ' ')
			list_truth_w.append(twords + '\n')

	path_result ='handwritten-text-recognition/output-KHATT-distorted/khatt/flor/default'
	if not os.path.exists(path_result):
		os.makedirs(path_result)

	f = codecs.open(path_result + 'c_reco_' + set + '.txt', 'w', 'utf-8')
	f.writelines(list_reco_c)
	f.close()

	f = codecs.open(path_result + 'c_truth_' + set + '.txt', 'w', 'utf-8')
	f.writelines(list_truth_c)
	f.close()

	f1 = codecs.open(path_result + 'w_reco_' + set + '.txt', 'w', 'utf-8')
	f1.writelines(list_reco_w)
	f1.close()

	f1 = codecs.open(path_result + 'w_truth_' + set + '.txt', 'w', 'utf-8')
	f1.writelines(list_truth_w)
	f1.close()
	#####################compute result CER%
	command3 = 'wer -a -e  ' + path_result + 'c_truth_' + set + '.txt' + ' ' + path_result + 'c_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_CER.txt'
	os.system(command3)
	#####################compute result WER%
	command2 = 'wer -a -e ' + path_result + 'w_truth_' + set + '.txt' + ' ' + path_result + 'w_reco_' + set + '.txt  >' + path_result + '/evaluate' + set + '_WER.txt'
	os.system(command2)


	


if __name__ == '__main__':
	train(150,32)
	 

	recognition_hard()
	