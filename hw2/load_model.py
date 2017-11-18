import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys

class Video_Caption_Generator():
  def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
    self.dim_image = dim_image
    self.n_words = n_words
    self.dim_hidden = dim_hidden
    self.batch_size = batch_size
    self.n_lstm_steps = n_lstm_steps
    self.n_video_lstm_step=n_video_lstm_step
    self.n_caption_lstm_step=n_caption_lstm_step

    # 指定用cpu跑
    with tf.device("/cpu:0"):
      # word Embedding, 每個word隨機開始，有dim_hidden個dimension
      self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
    #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

    self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

    # image feature的weight為一隨機變數(-.1 ~ .1)
    self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
    # image feature的bias為zero vecotr
    self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
    # word的weight為一隨機變數(-.1 ~ .1)????????????為什麼維度這樣設計
    self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')

    self.W_c = tf.Variable(tf.random_uniform([self.lstm2.state_size * 2, self.lstm2.state_size], -0.1,0.1), name='attention_W')
    self.b_c = tf.Variable(tf.zeros([self.lstm2.state_size]), name='attention_b')

    # 如果word有bias_init_vector就引用，沒又就用zero vecotr
    if bias_init_vector is not None:
      self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
    else:
      self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

  # 用在test
  def build_generator(self):
    video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
    video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

    video_flat = tf.reshape(video, [-1, self.dim_image])
    image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
    image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

    state1 = tf.zeros([1, self.lstm1.state_size])
    state2 = tf.zeros([1, self.lstm2.state_size])
    padding = tf.zeros([1, self.dim_hidden])

    generated_words = []

    probs = []
    embeds = []

    for i in range(0, self.n_video_lstm_step):

      with tf.variable_scope("LSTM1"):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        output1, state1 = self.lstm1(image_emb[:, i, :], state1)

      with tf.variable_scope("LSTM2"):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

      expand_encode_state2 = tf.expand_dims(state2, 2)
      if(i == 0):
        context_vector = expand_encode_state2
      else:
        context_vector = tf.concat([context_vector, expand_encode_state2], 2)

    for i in range(0, self.n_caption_lstm_step):
      if i == 0:
        with tf.device('/cpu:0'):
          tf.get_variable_scope().reuse_variables()
          current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

      with tf.variable_scope("LSTM1"):
        tf.get_variable_scope().reuse_variables()
        output1, state1 = self.lstm1(padding, state1)

      with tf.variable_scope("LSTM2"):
        tf.get_variable_scope().reuse_variables()
        expand_decode_state2 = tf.expand_dims(state2, 2)
        scores = tf.reduce_sum(tf.multiply(context_vector, expand_decode_state2), 1, keep_dims=True)
        a_t = tf.nn.softmax(scores)
        c_t  = tf.matmul(context_vector, tf.transpose(a_t, perm=[0,2,1]))
        c_t  = tf.squeeze(c_t, [2])
        state2_tld = tf.tanh(tf.matmul(tf.concat([state2, c_t], 1), self.W_c) + self.b_c)
        output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2_tld)
        # output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

      logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
      max_prob_index = tf.argmax(logit_words, 1)[0]
      generated_words.append(max_prob_index)
      probs.append(logit_words)

      with tf.device("/cpu:0"):
        current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
        current_embed = tf.expand_dims(current_embed, 0)

      embeds.append(current_embed)

    return video, video_mask, generated_words, probs, embeds


#=====================================================================================
# Global Parameters
#=====================================================================================

DATA_DIR = sys.argv[1]

video_test_feat_path = DATA_DIR + '/testing_data/feat'
video_peer_feat_path = DATA_DIR + '/peer_review/feat'

video_test_data_path = DATA_DIR + '/testing_id.txt'
video_peer_data_path = DATA_DIR + '/peer_review_id.txt'

output_test_path = sys.argv[2]
output_peer_path = sys.argv[3]

#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 25
n_frame_step = 80

n_epochs = 200
batch_size = 200
learning_rate = 0.001


def get_video_test_data(video_data_path, video_feat_path):
  video_data = pd.read_csv(video_data_path, header=None)
  video_data['video_path'] = video_data.apply(lambda row: row[0] +'.npy', axis=1)
  video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
  video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]

  unique_filenames = sorted(video_data['video_path'].unique())
  test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
  return test_data

def test(model_path='./model'):
  ixtoword = pd.Series(np.load('./mapping/ixtoword.npy').tolist())

  bias_init_vector = np.load('./mapping/bias_init_vector.npy')

  model = Video_Caption_Generator(
      dim_image=dim_image,
      n_words=len(ixtoword),
      dim_hidden=dim_hidden,
      batch_size=batch_size,
      n_lstm_steps=n_frame_step,
      n_video_lstm_step=n_video_lstm_step,
      n_caption_lstm_step=n_caption_lstm_step,
      bias_init_vector=bias_init_vector)

  video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

  sess = tf.InteractiveSession()

  saver = tf.train.Saver()
  saver.restore(sess, model_path)

  def outputRes(id_path, feat_path, output_path):
    test_data = get_video_test_data(id_path, feat_path)
    test_videos = test_data['video_path'].unique()

    test_output_txt_fd = open(output_path, 'w')
    for idx, video_feat_path in enumerate(test_videos):
      print(video_feat_path)
      video_feat = np.load(video_feat_path)[None,...]

      if video_feat.shape[1] == n_frame_step:
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
      else:
        continue

      generated_word_index, probs = sess.run([caption_tf, probs_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

      generated_words = ixtoword[generated_word_index]

      punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
      generated_words = generated_words[:punctuation]

      generated_sentence = ' '.join(generated_words)
      generated_sentence = generated_sentence.replace('<bos> ', '')
      generated_sentence = generated_sentence.replace(' <eos>', '')
      print(generated_sentence,'\n')
      test_output_txt_fd.write(video_feat_path.replace(DATA_DIR, '').replace('/peer_review/feat/', '').replace('/testing_data/feat/', '').replace('.npy', ''))
      test_output_txt_fd.write(',' + generated_sentence + '\n')

  outputRes(id_path=video_test_data_path, feat_path=video_test_feat_path, output_path=output_test_path)
  outputRes(id_path=video_peer_data_path, feat_path=video_peer_feat_path, output_path=output_peer_path)

test()
