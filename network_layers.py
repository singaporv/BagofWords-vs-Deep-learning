import numpy as np
import scipy.ndimage
import os,time

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	for L in vgg16_weights[:35]:
		if (L[0] == 'conv2d'):
			weight = L[1]
			bias = L[2]
			x = multichannel_conv2d(x,weight,bias)
		if (L[0] == 'relu'):
			x  = relu(x)
		if (L[0] == 'maxpool2d'):
			size = L[1]
			# print (x.shape)
			x = max_pool2d(x,size)
		if (L[0] == 'linear'):
			weight = L[1]
			bias = L[2]
			x = linear(x,weight,bias)




	feat = x
	return feat


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.
	
	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)
	
	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	list_2 = []
	for i in range(weight.shape[0]):
		list_1 = []
		for j in range(weight.shape[1]):
			w = np.flip(weight[i, j, :, :])
			convo = scipy.ndimage.convolve(x[:, :, j], w, mode = 'constant', cval = 0.0)
			list_1.append(convo)
		f = sum(list_1) + bias[i]
		list_2.append(f)

	y = np.dstack(list_2)
	return y
	# convo = fliplr(convo)
	# convo = flipud(convo)


	

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	y = np.maximum(x,0)
	return y

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field
	
	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	H,W,input_dim = x.shape
	print (size)
	list_3 = []
	print (x.shape)
	for i in range(x.shape[2]):
		y = x[:,:,i].reshape(H//size,size, W//size, size).max(axis= 1).max(axis=2)
		list_3.append(y)

	y = np.dstack(list_3)
	return y
	

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	x = x.flatten()
	y = np.matmul(x, np.transpose(W)) + b
	return y



