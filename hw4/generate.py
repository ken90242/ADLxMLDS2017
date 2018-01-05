import os
from os.path import join, isfile
import sys
import h5py
import re
import random
import json
import tensorflow as tf
import numpy as np
import scipy.misc
import yaml
import pickle as pkl
import model
import time
from Utils import image_processing, skipthoughts

def main():
	model_parameters = yaml.load(open('Config.yaml', 'r'))

	model_parameters['batch_size'] = 5 # n_images

	caption_path = sys.argv[1]

	gan = model.GAN(model_parameters)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, 'model')
	
	input_tensors, outputs = gan.build_generator()
	test_ids=[]
	with open( caption_path ) as f:
		for line in f:
			test_ids.append(line.split(',')[0])	

	with open( caption_path ) as f:
		captions = f.read().split('\n')
		captions = [re.sub('\d+,', '', cap) for cap in captions]
		captions = [re.sub('(hair|eyes)(\s+)', '\g<1> and ', cap) for cap in captions]

	captions = [cap for cap in captions if len(cap) > 0]
	caption_vectors = skipthoughts.encode(skipthoughts.load_model(), captions)
	caption_image_dic = {}
	z_noise_arr = pkl.load(open('noise.pkl', 'rb'))

	for cn, caption_vector in enumerate(caption_vectors):

		caption_images = []
		# z_noise = np.random.uniform(-1, 1, [model_parameters['batch_size'], model_parameters['z_dim']])
		z_noise = z_noise_arr[cn]
		caption = [ caption_vector[0:2400] ] * model_parameters['batch_size']
		
		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )
		
		caption_images = [gen_image[i,:,:,:] for i in range(0, model_parameters['batch_size'])]
		caption_image_dic[ cn ] = caption_images

	for f in os.listdir( join('samples')):
		if os.path.isfile(f):
			os.unlink(join('samples/' + f))

	for i, cn in enumerate(range(0, len(caption_vectors))):
		test_id = test_ids[i]
		for j, im in enumerate( caption_image_dic[ cn ] ):
			scipy.misc.imsave( join('samples/sample_{}_{}.jpg'.format(test_id, j + 1)) , im)
			print("Generate sample_{}_{}.jpg in samples/".format(test_id, j + 1))


if __name__ == '__main__':
	main()
