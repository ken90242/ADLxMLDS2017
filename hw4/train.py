import tensorflow as tf
import numpy as np
import model
import argparse
import pickle as pkl
from os.path import join
from Utils import image_processing
import scipy.misc
import random
import os
import shutil
import yaml

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, default="Data",
					   help='Data Directory')

	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--train_name', type=str, default='default',
                       help='Append nick name')

	args = parser.parse_args()

	logf = open('logs/logs_{}.txt'.format(args.train_name), 'a')
	logf.write('d_loss\tg_loss\n')


	raw_real_images, raw_captions = pkl.load(open(join(args.data_dir, 'dataset.pkl'), 'rb'))

	MAX_EPOCH = 300
	TRAIN_LEN = len(raw_captions)
	
	# model parameters
	model_parameters = yaml.load(open('Config.yaml', 'r'))
	gan = model.GAN(model_parameters)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	
	# beta1(momentum) = 0.5
	d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss['d_loss'], var_list=variables['d_vars'])
	g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss['g_loss'], var_list=variables['g_vars'])
	
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	saver = tf.train.Saver(max_to_keep=None)
	if args.resume_model:
		saver.restore(sess, args.resume_model)
	
	for i in range(MAX_EPOCH):
		batch_no = 0
		batch_size = model_parameters['batch_size']
		while batch_no*batch_size < TRAIN_LEN:
			real_images, wrong_images, caption_vectors, z_noise = get_training_batch(TRAIN_LEN, raw_real_images, raw_captions, batch_no, input_tensors, model_parameters, args.data_dir)

			feed_dict = {
				input_tensors['t_real_image']: real_images,
				input_tensors['t_wrong_image']: wrong_images,
				input_tensors['t_real_caption']: caption_vectors,
				input_tensors['t_z']: z_noise,
			}

			# DISCR UPDATE
			check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
			_, d_loss, generated_images, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
				feed_dict=feed_dict)
			
			# GEN UPDATE TWICE, to make sure d_loss does not go to 0
			for _ in range(2):
				# 隨機調換真假的圖片
				if(random.random() < 0.5):
					feed_dict = {
						input_tensors['t_real_image']: wrong_images,
						input_tensors['t_wrong_image']: real_images,
						input_tensors['t_real_caption']: caption_vectors,
						input_tensors['t_z']: z_noise,
					}
				_, g_loss, generated_images = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict=feed_dict)

			
			print('Real: {:.2f} / Wrong_cap :{:.2f} / Fake :{:.2f}'.format(d1, d2, d3))
			# avg_batch_imgLen: TRAIN_LEN/batch_size
			print("[{}-{}]\td_loss:{:.2f} / g_loss:{:.2f}\n".format(i, batch_no, d_loss, g_loss))
			logf.write('{}\t{}\n'.format(d_loss, g_loss))


			batch_no += 1

		print("# Save Img/ep")
		save_for_vis(args.data_dir, args.train_name, feed_dict[input_tensors['t_real_image']], generated_images, i)

		if i % 2 == 0:
			print("# Save Model/2 ep")
			save_path = saver.save(sess, "{}/model_{}_epoch_{}.ckpt".format(args.resume_model, args.train_name, i))

	logf.close()

def save_for_vis(data_dir, nick_name, real_images, generated_images, epoch):
	combinated_fake_image = np.zeros( (512, 512, 3), dtype=np.uint8)	
	combinated_row_fake_image = np.zeros( (64, 512, 3), dtype=np.uint8)	

	for i in range(0, real_images.shape[0]):
		for a in range(8):
			for b in range(8):
				t = generated_images[a * 8 + b]
				if(b==0):
					combinated_row_fake_image = t
				else:
					combinated_row_fake_image = np.concatenate((combinated_row_fake_image, t), axis=1)
			if(a==0):
				combinated_fake_image = combinated_row_fake_image
			else:
				combinated_fake_image = np.concatenate((combinated_fake_image, combinated_row_fake_image), axis=0)

	scipy.misc.imsave(join(data_dir, 'samples_process/epoch_{}_{}_fake.jpg'.format(nick_name, epoch)), combinated_fake_image)


def get_training_batch(TRAIN_LEN, raw_real_images, raw_captions, batch_no, input_tensors, model_parameters, data_dir):
	(batch_size, image_size, channels, z_dim, caption_vector_length) = \
		(model_parameters[x] for x in ['batch_size', 'image_size', 'channels', 'z_dim', 'caption_vector_length'])

	real_images = np.zeros((batch_size, image_size, image_size, channels))
	wrong_images = np.zeros((batch_size, image_size, image_size, channels))
	caption_vectors = np.zeros((batch_size, caption_vector_length))

	train_seq = random.sample(range(TRAIN_LEN), TRAIN_LEN)

	for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
		idx = i % batch_size
		seq = train_seq[idx]
		caption_vectors[idx, :] = raw_captions[seq][0:caption_vector_length]

		real_images[idx, :,:,:] = np.fliplr(raw_real_images[seq])\
																if random.random() > 0.5 else raw_real_images[seq]

		# Improve this selection of wrong image
		wrong_image_id = random.randint(0, 33430)
		wrong_image_file =  join('faces/' + str(wrong_image_id) + '.jpg')
		wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
		wrong_images[idx, :,:,:] = wrong_image_array

	# 均勻的隨機取數
	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

	# Normalize between [ -1, 1 ]
	# r_min = real_images.min(axis=(0,1), keepdims=True)
	# r_max = real_images.max(axis=(0,1), keepdims=True)
	# real_images = 2 * ((real_images - r_min) / (r_max - r_min)) - 1
	# print(wrong_images[0][0])

	# a=[(-1<=x).all() and (x<=1).all() for x in wrong_images]
	# print('pass' if any(x == True for x in a) else 'fail')

	return real_images, wrong_images, caption_vectors, z_noise

if __name__ == '__main__':
	main()
