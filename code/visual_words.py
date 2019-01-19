import numpy as np
from multiprocessing import Pool
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import math

def extract_filter_responses(image):
	# if the image is gray (just h and w will come in image shape) -- then this will convert in into image of 3 equal channels.
	if (len(image.shape)<3):
		image = image[:,:]
	# if the image is having more than 3 channels then this will delete the additional channels.
	elif (image.shape[2])>3:
		image = image[:,:,:3]
	
	image_collection = image[:,:,0], image[:, :, 1], image[:, :, 2]
	scale_collection = [1,2,4,8,8*math.sqrt(2)]
	image = skimage.color.rgb2lab(image)

	final = []

	for i in range(len(scale_collection)):
		for j in range(len(image_collection)):
			final.append(scipy.ndimage.gaussian_filter(image_collection[j], sigma = scale_collection[i]))
		
		for j in range(len(image_collection)):
			final.append(scipy.ndimage.gaussian_laplace(image_collection[j], sigma = scale_collection[i]))
			
		for j in range(len(image_collection)):
			final.append(scipy.ndimage.gaussian_filter(image_collection[j], sigma = scale_collection[i], order = [0,1]))

		for j in range(len(image_collection)):
			final.append(scipy.ndimage.gaussian_filter(image_collection[j], sigma = scale_collection[j], order = [1,0]))

	# this function stacks the elements onto each other to create the array 
	return np.dstack(final)
	



def get_visual_words(image,dictionary):
	filter_response = extract_filter_responses(image)
	# print (filter_response.shape)
	h,w,d =  filter_response.shape
	filter_response = filter_response.reshape(h*w, d)
	# print (filter_response.shape)
	
	distance = scipy.spatial.distance.cdist(filter_response, dictionary)
	# print (len(distance))

	# print (distance[0])
	min_dist_list = []
	for i in  distance:
		min_dist_list.append(np.argmin(i))
	wordmap = np.asarray(min_dist_list).reshape(h,w)
	# print (wordmap.shape)
	plt.imshow(wordmap, cmap = 'hsv')
	plt.show()
	return wordmap
	

	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	# print (dictionary.shape)


def compute_dictionary_one_image(args): 
	index, img_path, alpha = args
	# print (index,img_path, alpha)

	image = skimage.io.imread(img_path)
	image = image.astype('float')/255

	filter_response = extract_filter_responses(image)
	# print (filter_response.shape)
	
	arr_1 = []
	for i in range(alpha):
		random_response_height = random.randint(0, filter_response.shape[0]-1)
		random_response_width = random.randint(0, filter_response.shape[1]-1)
		arr_1.append(filter_response[random_response_height, random_response_width, :])
	arr_1 = np.asarray(arr_1)

	np.save('../temp/%d'%index, arr_1)




def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	# to load the compressed (.npz file)
	train_data = np.load("../data/train_data.npz")
	labels_pool =  train_data['labels']
	images_pool =  train_data['image_names']

	# print (images_pool.shape)
	# print (labels_pool.shape)

	# alpha is the value of the random pixels from the image that are to be picked
	alpha = 75

	#this list will contain index, name of the image and value of alpha
	arg_list = []
	for i in range(images_pool.shape[0]):
		arg_list.append((i, '../data/' + images_pool[i][0], alpha))

	# these four commands are used to use multiple cores for creating the list
	p = Pool(processes=num_workers)
	p.map(compute_dictionary_one_image, arg_list)
	p.close()
	p.join()

	# this is to load the files 
	arr_2 = []
	for k in range(images_pool.shape[0]):
		 arr_2.append(np.load('../temp/%d.npy'%k))

	# fist list is converted into array and then it is reshaped
	fr = np.asarray(arr_2).reshape(alpha*images_pool.shape[0],60)
	# print (fr.shape)



	kmeans = sklearn.cluster.KMeans(n_clusters=100,n_jobs=num_workers).fit(fr)
	dictionary = kmeans.cluster_centers_
	np.save('dictionary', dictionary)


