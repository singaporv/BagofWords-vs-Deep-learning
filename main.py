import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

if __name__ == '__main__':

	# read the number of cores
	num_cores = util.get_num_CPU()
	print (num_cores)
	# path_img = "../data/baseball_field/sun_afdicykxalgviezn.jpg"

	# skimage.io function for reading the image
	# image = skimage.io.imread(path_img)
	# plt.imshow(image, cmap = 'hsv')
	# plt.show()
	# print (image.shape)

	# numpy array to convert array into a specific datatype
	# image = image.astype('float')/255


	# filter_responses = visual_words.extract_filter_responses(image)
	# print(filter_responses.shape)
	# util.display_filter_responses(filter_responses)
	# visual_words.compute_dictionary(num_workers=num_cores)	
	# dictionary = np.load('dictionary.npy')
	# wordmap = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(wordmap, filename)
	# dict_size = 100
	# layer_num = 3
	# visual_recog.get_feature_from_wordmap(wordmap,dict_size)
	# histogram = visual_recog.get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size)
	
	# visual_recog.build_recognition_system(num_workers=num_cores)

	# conf = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	# print(conf)
	# print(np.diag(conf).sum()/conf.sum())

	# vgg16 = torchvision.models.vgg16(pretrained=True).double()
	# vgg16.eval()
	# deep_recog.get_image_feature((None, None, vgg16))

	# deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	# conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	# print(conf)
	# print(np.diag(conf).sum()/conf.sum())


