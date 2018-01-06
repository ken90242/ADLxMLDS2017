import tensorflow as tf
import tensorflow.contrib.layers as ly
from Utils import ops

def conv_cond_concat(x, y):
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
		   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv
# http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
# def minibatch(input, num_kernels=100, kernel_dim=5): # input: (64, 8192)
# 	# (64, 15)
# 	x = linear(input, num_kernels * kernel_dim)
# 	# (64, 5, 3)
# 	activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
# 	# diffs: (64, 5, 3, 64) = (64, 5, 3, 1) - (1, 5, 3, 64)
# 	# *tf.transpose: (5, 3, 64)
# 	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
# 	# (64, 5, 64)
# 	abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
# 	# (64, 5)
# 	minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
# 	# (64, 8197)
# 	return tf.concat([x, minibatch_features], axis=1)

# https://github.com/tdrussell/IllustrationGAN/blob/master/custom_ops.py
# 此篇的輸入為32*32*3, default:num_kernels=100, kernel_dim=5
def minibatch(input_layer, num_kernels, kernel_dim): # input: (64, 8192)
	# (64, 15)
	batch_size = input_layer.shape[0]
	num_features = input_layer.shape[1]

	W = tf.get_variable("minibatch_w", [num_features, num_kernels * kernel_dim],\
		tf.float32, tf.random_normal_initializer(stddev=0.02))
	b = tf.get_variable("minibatch_b", [num_kernels], initializer=tf.constant_initializer(0.0))

	activation = tf.matmul(input_layer, W)
	# (64, 5, 3)
	activation = tf.reshape(activation, (-1, num_kernels, kernel_dim))
	# diffs: (64, 5, 3, 64) = (64, 5, 3, 1) - (1, 5, 3, 64)
	# *tf.transpose: (5, 3, 64)
	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
	# (64, 5, 64)
	abs_diffs = tf.reduce_sum(tf.abs(diffs), reduction_indices=[2])
	# (64, 5)
	minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), reduction_indices=[2])
	# (64, 8197)
	return minibatch_features + b

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.random_normal_initializer(stddev=stddev))
		
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		return deconv

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		
		return tf.matmul(input_, matrix) + bias

class GAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, options):
		self.options = options

		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')

		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')


	def build_model(self):
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3 ], name = 'real_image')
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3 ], name = 'wrong_image')
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

		fake_image = self.generator(t_z, t_real_caption)

		# real image, right text
		disc_real_image, disc_real_image_logits = self.discriminator(t_real_image, t_real_caption)
		# real image, wrong text = image is REAL, caption DOESN'T FIT!
		disc_wrong_image, disc_wrong_image_logits = self.discriminator(t_wrong_image, t_real_caption, reuse=True)
		# fake image, right text
		disc_fake_image, disc_fake_image_logits = self.discriminator(fake_image, t_real_caption, reuse=True)

		# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image))) # original
		# g_loss = -tf.reduce_mean(disc_fake_image_logits) #get rid of log
		g_loss = 0.5 * tf.reduce_mean((disc_fake_image_logits - tf.ones_like(disc_fake_image)) ** 2) # least-square

		'''
		[tf.nn.sigmoid_cross_entropy_with_logits]
			(let x = logits, z = labels)
			=	z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
			=	max(x, 0) - x * z + log(1 + exp(-abs(x)))

			問：d_loss... 這些的物理意義？？
		'''
		# d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
		# d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits, labels=tf.zeros_like(disc_wrong_image)))
		# d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

		# least-square
		d_loss1 = 0.5 * tf.reduce_mean((disc_real_image_logits - tf.ones_like(disc_real_image)) ** 2)
		d_loss2 = 0.5 * tf.reduce_mean((disc_wrong_image_logits - tf.zeros_like(disc_wrong_image)) ** 2)
		d_loss3 = 0.5 * tf.reduce_mean((disc_fake_image_logits - tf.zeros_like(disc_fake_image)) ** 2)

		d_loss = d_loss1 + d_loss2 + d_loss3
		

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		# Loss function using L2 Regularization
		regularizers = 0
		for d_weight in d_vars:
			regularizers += tf.nn.l2_loss(d_weight)
		d_loss = tf.reduce_mean(d_loss + 0.01 * regularizers)
    


		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits
		}

		return input_tensors, variables, loss, outputs, checks

	def build_generator(self):
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		fake_image = self.sampler(t_z, t_real_caption)

		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# # Sample Images for a text embedding
	def sampler(self, t_z, t_text_embedding):
		tf.get_variable_scope().reuse_variables()

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		reduced_text_embedding = lrelu( linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding], axis=1)
		z_ = linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))

		h1 = deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))

		h2 = deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))

		h3 = deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))

		h4 = deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		return (tf.tanh(h4)/2. + 0.5)

	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	'''
	# Generator

		[Input]
			t_z = (batch_size, 100)
			t_text_embedding = (batch_size, 2400)

		[Process]
		* gf_dim : Number of conv in the first layer generator 64 *

			reduce_t_text = (batch_size, 256)
			z_concat = concat(t_z, reduce_t_text)

		[Output]
			h4 = (batch_size, 64, 64, 3)

	'''
	def generator(self, t_z, t_text_embedding):

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		reduced_text_embedding = lrelu( linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding], axis=1)
		z_ = linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16, 'g_h0_lin')

		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))

		h1 = deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))

		h2 = deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))

		h3 = deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))

		h4 = deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		# Normalize between [ 0.5, 1 ]
		return (tf.tanh(h4) / 2. + 0.5)

	'''
	# Discriminator

		[Input]
			image = (batch_size, 64, 64, 3)
			t_text_embedding = (batch_size, 2400)

		[Process]
		* df_dim : Number of conv in the first layer discriminator 64 *

			reduce_t_text = (batch_size, 256)
			z_concat = concat(t_z, reduce_t_text)

		[Output]
			h4 = (batch_size, 1)

	'''
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			# tf.get_variable_scope().reuse_variables()
			with tf.variable_scope(tf.get_variable_scope()) as scope:
				scope.reuse_variables()
				h0 = lrelu(conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) # [64, 32, 32, 64]
				h1 = lrelu(self.d_bn1(conv2d(h0, self.options['df_dim'] * 2, name = 'd_h1_conv'))) # [64, 16, 16, 128]
				h2 = lrelu(self.d_bn2(conv2d(h1, self.options['df_dim'] * 4, name = 'd_h2_conv'))) # [64, 8, 8, 256]
				h3 = lrelu(self.d_bn3(conv2d(h2, self.options['df_dim'] * 8, name = 'd_h3_conv'))) # [64, 4, 4, 512]

				# ADD TEXT EMBEDDING TO THE NETWORK
				reduced_text_embeddings = lrelu(linear(t_text_embedding, self.options['t_dim'], 'd_embedding')) # [64, 256]

				reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1) # [64, 1, 256]

				reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2) # [64, 1, 1, 256]

				tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings') # [64, 4, 4, 256]
				h3_concat = tf.concat( [h3, tiled_embeddings], axis=3, name='h3_concat')
				h3_new = lrelu( self.d_bn4(conv2d(h3_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name='d_h3_conv_new'))) # [64, 4, 4, 512]
				
				# h4 = linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin') # [64, 1]

				h4 = tf.reshape(h3_new, [self.options['batch_size'], -1]) # [64, 5] -> [64, 1]
				h4 = linear(minibatch(h4, num_kernels=150, kernel_dim=8), 1, 'd_h3_lin')
		
				return tf.nn.sigmoid(h4), h4
		else:
			h0 = lrelu(conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) # [64, 32, 32, 64]
			h1 = lrelu(self.d_bn1(conv2d(h0, self.options['df_dim'] * 2, name = 'd_h1_conv'))) # [64, 16, 16, 128]
			h2 = lrelu(self.d_bn2(conv2d(h1, self.options['df_dim'] * 4, name = 'd_h2_conv'))) # [64, 8, 8, 256]
			h3 = lrelu(self.d_bn3(conv2d(h2, self.options['df_dim'] * 8, name = 'd_h3_conv'))) # [64, 4, 4, 512]

			# ADD TEXT EMBEDDING TO THE NETWORK
			reduced_text_embeddings = lrelu(linear(t_text_embedding, self.options['t_dim'], 'd_embedding')) # [64, 256]

			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1) # [64, 1, 256]

			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2) # [64, 1, 1, 256]

			tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings') # [64, 4, 4, 256]
			h3_concat = tf.concat( [h3, tiled_embeddings], axis=3, name='h3_concat')
			h3_new = lrelu( self.d_bn4(conv2d(h3_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name='d_h3_conv_new'))) # [64, 4, 4, 512]
			# h4 = linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin') # [64, 1]

			h4 = tf.reshape(h3_new, [self.options['batch_size'], -1]) # [64, 8192]
			h4 = linear(minibatch(h4, num_kernels=150, kernel_dim=8), 1, 'd_h3_lin') # [64, 1]
			# print(h4)

			return tf.nn.sigmoid(h4), h4

