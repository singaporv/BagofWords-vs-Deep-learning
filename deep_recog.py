import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from multiprocessing import Pool
import sklearn.metrics
import scipy.spatial.distance



def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	# np.save('trained_system_deep', ______)


	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	# print ('dictionary size')
	# print (dictionary.shape)
	labels_pool =  train_data['labels']
	images_names =  train_data['image_names']


	'''These below lines will train the data. It can be commented out once model has been trained. 
	image_names_list = []
	for i in range(images_names.shape[0]):
		image_names_list.append((i, '../data/' + images_names[i][0],vgg16))

	p = Pool(processes=num_workers)
	p.map(get_image_feature, image_names_list)
	p.close()
	p.join()


	after maodel has trained -- got to visual words and save the copy'''
	arr_5 = []
	for k in range(images_names.shape[0]):
		 arr_5.append(np.load('../temp1/%d.npy'%k))

	features = np.asarray(arr_5)

	np.savez_compressed('trained_system_deep', features = features, 
		labels = labels_pool)


def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	trained_system= np.load("trained_system_deep.npz")
	test_data = np.load("../data/test_data.npz")

	labels_pool =  test_data['labels']
	images_names =  test_data['image_names']

	features = trained_system["features"]
	labels = trained_system["labels"]


	image_names_list = []
	for i in range(images_names.shape[0]):
		image_names_list.append(('../data/' + images_names[i][0], features, vgg16))

	p = Pool(processes=num_workers)
	indexes = p.map(evaluate_single_img, image_names_list)
	p.close()
	p.join()

	print(indexes)

	prediction = []
	for index in indexes:
		prediction.append(labels[index])
	prediction = np.asarray(prediction)

	return sklearn.metrics.confusion_matrix(labels_pool, prediction)



def evaluate_single_img(args):

	image_path, train_features, vgg16 = args
	print (image_path)
	image = skimage.io.imread(image_path)
	# print (image.shape)
	x = preprocess_image(image)
	# print (x.shape)

	''' these below lines will be required for non-pytorch processing
	vgg16_weights = util.get_VGG16_weights()
	feat = network_layers.extract_deep_feature(x, vgg16_weights)
	'''
	# print (feat)


	# if pytorch
	top_layers = torch.nn.Sequential(*list(vgg16.children())[0])
	fc7 = torch.nn.Sequential(*list(vgg16.children())[1][:5])	
	feat = fc7(top_layers(x).flatten()).detach().numpy()

	# feature = get_image_feature(i, image_path, vgg16)
	distance =  distance_to_set(feat, train_features)
	return distance


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	# ----- TODO -----
	

	image = image.astype('float')/255

	if (len(image.shape)<3):
		image = image[:,:]
	# if the image is having more than 3 channels then this will delete the additional channels.
	elif (image.shape[2])>3:
		image = image[:,:,:3]

	image = skimage.transform.resize(image, (224,224), anti_aliasing = True)
	mean = [0.485,0.456,0.406]
	std = [0.229,0.224,0.225]

	image = (image - mean)/std
	# print (image.shape)

	#below lines could be commented for non-pytorch processing
	# convert function
	convert = torchvision.transforms.ToTensor()
	# to convert in tensor
	image = convert(image).unsqueeze(0)

	return image


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	''' this line will be used alternatively - to test the non-pytorch processing
	image_path = "../data/desert/sun_aaqyzvrweabdxjzo.jpg"
	'''

	i, image_path, vgg16 = args
	print (image_path)
	image = skimage.io.imread(image_path)
	# print (image.shape)
	x = preprocess_image(image)
	# print (x.shape)

	''' these below lines will be required for non-pytorch processing
	vgg16_weights = util.get_VGG16_weights()
	feat = network_layers.extract_deep_feature(x, vgg16_weights)
	'''
	# print (feat)


	# if pytorch
	top_layers = torch.nn.Sequential(*list(vgg16.children())[0])
	fc7 = torch.nn.Sequential(*list(vgg16.children())[1][:5])	

	feat = fc7(top_layers(x).flatten())

	feat = feat.detach().numpy()
	print(feat.shape) # 4096

	np.save('../temp1/%d'%i, feat)

	# i,image_path,vgg16 = args

	# ----- TODO -----




def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	distance = scipy.spatial.distance.cdist(np.expand_dims(feature, 0), train_features)


	return np.argmin(distance)