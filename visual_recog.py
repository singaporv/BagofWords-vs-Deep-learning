import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import cv2
import matplotlib as plt
from multiprocessing import Pool
import skimage.io
import sklearn.metrics

	


def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	# print ('dictionary size')
	# print (dictionary.shape)
	labels_pool =  train_data['labels']
	images_names =  train_data['image_names']

	K = 100
	layer_num = 3

	image_names_list = []
	for i in range(images_names.shape[0]):
		image_names_list.append(('../data/' + images_names[i][0], dictionary, layer_num, K))


	p = Pool(processes=num_workers)
	features = p.starmap(get_image_feature, image_names_list)
	p.close()
	p.join()
	features = np.asarray(features)

	np.savez_compressed('trained_system', features = features, 
		labels = labels_pool, dictionary= dictionary, SPM_layer_num= layer_num)


def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	trained_system= np.load("trained_system.npz")
	test_data = np.load("../data/test_data.npz")

	labels_pool =  test_data['labels']
	images_names =  test_data['image_names']

	K = 100

	features = trained_system["features"]
	labels = trained_system["labels"]
	dictionary = trained_system["dictionary"]
	layer_num = trained_system["SPM_layer_num"]


	image_names_list = []
	for i in range(images_names.shape[0]):
		image_names_list.append(('../data/' + images_names[i][0], ))

	p = Pool(processes=num_workers)
	indexes = p.starmap(evaluate_single_img, image_names_list)
	p.close()
	p.join()

	print(indexes)

	prediction = []
	for index in indexes:
		prediction.append(labels[index])
	prediction = np.asarray(prediction)

	return sklearn.metrics.confusion_matrix(labels_pool, prediction)

def evaluate_single_img(file_path,dictionary,layer_num,K, histograms):

	print (file_path)
	image = skimage.io.imread(file_path)
	image = image.astype('float')/255

	wordmap = visual_words.get_visual_words(image, dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	distances =  distance_to_set(feature, histograms)

	return np.argmax(distances)

def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''

	print (file_path)
	image = skimage.io.imread(file_path)
	image = image.astype('float')/255

	wordmap = visual_words.get_visual_words(image, dictionary)
	features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)

	return features

def distance_to_set(word_hist, histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''

	sim = []
	for i in range(histograms.shape[0]):
		arr = np.sum(np.minimum(word_hist, histograms[i]))
		sim.append(arr)

	# this list will be of size K (Containing sum of minimums)
	sim = np.asarray(sim)
	return sim

def get_histogram_function(wordmap, pixels, dict_size):
	'''
	This function normalizes the histogram
	'''


	hist, bins = np.histogram(wordmap, bins = dict_size)

	hist = hist/pixels
	return hist


def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	h,w =  wordmap.shape
	pixels = h*w
	hist = get_histogram_function(wordmap, pixels, dict_size)
	# print (hist.shape)
	# print (hist)

	return hist


def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	h, w = wordmap.shape
	pixels = h*w

	hist_list = []
	#L0
	hist = get_histogram_function(wordmap,pixels,dict_size)
	hist_list.append(hist)
	# print (hist_list)

	list_3 = []

	for i in range(1, layer_num):
		output_1 = np.array_split(wordmap,2**i,axis=0)
		for sub_matrix in output_1:
			list_3.append(np.array_split(sub_matrix,2**i,axis=1))

	list_3 = np.asarray(list_3)
	for row in list_3:
		for col in row:
			hist_list.append(get_histogram_function(col,pixels,dict_size))
	# print (len(np.asarray(hist_list)))
	# print (np.sum(np.asarray(hist_list)))

	weight_list = [1/4,1/4, 1/4, 1/4, 1/4]
	for i in range(5,21):	
		weight_list.append(1/2)
	
	hist_list = np.asarray(hist_list)

	for i in range(len(hist_list)):
		hist_list[i] = hist_list[i]* weight_list[i]

	# print (np.sum(hist_list))
	histogram = np.concatenate(hist_list)
	return histogram
