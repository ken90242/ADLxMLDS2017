import re
import os
import sys
import time
import tensorflow as tf
import pandas as pd
import numpy as np
from random import randint
from keras.preprocessing import sequence

class Video_Caption_Generator():
  def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
    self.dim_image = dim_image
    self.n_words = n_words
    self.dim_hidden = dim_hidden
    self.batch_size = batch_size
    self.n_lstm_steps = n_lstm_steps
    self.n_video_lstm_step=n_video_lstm_step
    self.n_caption_lstm_step=n_caption_lstm_step

    # 用cpu跑
    with tf.device("/cpu:0"):
      # word Embedding, 每個word隨機，shape = (n_words, dim_hidden)
      self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
    #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')
    
    # ??????????????
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

  # 用在train
  def build_model(self):
    video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
    video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

    caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
    caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

    video_flat = tf.reshape(video, [-1, self.dim_image])
    image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
    # video被extraction為batch_size * n_lstm_steps(timesteps) * dim_hidden的dimension
    image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

    state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
    state2 = tf.zeros([self.batch_size, self.lstm2.state_size])

    # logit_words = tf.zeros([self.batch_size, self.n_words])

    padding = tf.zeros([self.batch_size, self.dim_hidden])
    # flip_sampling = tf.placeholder(tf.float32, shape=())

    probs = []
    loss = 0.0
  #####################  Attention Model #########################
    context_vector = tf.zeros([self.batch_size, self.lstm1.state_size, self.n_video_lstm_step])

    ###################  Encoding Stage ###################
    for i in range(0, self.n_video_lstm_step):

      with tf.variable_scope("LSTM1"):
        if i > 0:
          # 沿用上次的weights
          tf.get_variable_scope().reuse_variables()
        # 取出當前的timestep的batch和dim value
        # 再搭配state1輸入進lstm1
        output1, state1 = self.lstm1(image_emb[:,i,:], state1)

      with tf.variable_scope("LSTM2"):
        if i > 0:
          # 沿用上次的weights
          tf.get_variable_scope().reuse_variables()
        # t1 = [[1, 2, 3], [4, 5, 6]]
        # t2 = [[7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

        # (tensor t3 with shape [2, 3])
        # (tensor t4 with shape [2, 3])
        # tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
        # tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
        garbage, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)


      expand_encode_state2 = tf.expand_dims(state2, 2)

      if(i == 0):
        context_vector = expand_encode_state2
      else:
        context_vector = tf.concat([context_vector, expand_encode_state2], 2)

    ############################# Decoding Stage ######################################
    for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
      # 這裏是一次看一排的batch_size
      # 以shape(3,4,5)為例，它會分成4次，每次看(3,5)
      with tf.device("/cpu:0"):
        # 在第i個timestep下，將所有batch的sentence取出，並依照這句sentence裡的word一個一個去查詢他們的word_embedding vec
        # caption[:, i] = (200, 1); current_embed = (200, 500)
        # 這裏是將上一句取出，如：A man is talking
        # 在'A'的階段會取出'<bos>'，在'man'的階段會取出'A'，在'is'的階段會取出'man'...等
        # if(i == 0):
        ground_truth = caption[:, i]
        # prev_output = tf.argmax(logit_words, axis=1)

        # sample_result = [ground_truth, prev_output][np.random.choice(2, p=[tf.to_float(flip_sampling), 1 - tf.to_float(flip_sampling)])]
        current_embed = tf.nn.embedding_lookup(self.Wemb, ground_truth)

        # else:
          # current_embed = tf.nn.embedding_lookup(self.Wemb, output2)

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

      # 't2' is a tensor of shape [2, 3, 5], expand_dims補1(是"shape"!!)
      # shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
      # shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]

      # 建立onehot_labels
      # 為什麼是 i + 1 ? 因為輸入的每個caption都會在caption前面先加一個'<bos>'
      labels = tf.expand_dims(caption[:, i+1], 1)
      # tf.range(a,b,c) = [ a , a+c , a+2c , ... , b ]
      # 為什麼要做tf.range？單純是因為expand_dims要接受一個shape，而tf.range可以產生所需shape的array
      indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
      # concated(200, 2) = labels(200, 1) + indices(200, 1)
      # 為什麼要有indices及concated？單純為了符合sparse_to_dense的矩陣輸入格式
      concated = tf.concat([indices, labels], 1)

      # x = tf.constant([1, 4])
      # y = tf.constant([2, 5])
      # z = tf.constant([3, 6])
      ## [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
      # tf.stack([x, y, z])
      ## [[1, 2, 3], [4, 5, 6]]
      # tf.stack([x, y, z], axis=1)

      # tf.stack[...] = [ batch_size, n_words ]
      # args: input_matrix / output_shape / sparse_value / default_value
      ## If sparse_indices is an n by d matrix, then for each i in [0, n)
      # dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
      onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
      # 比較差異，並用caption_mask遮蔽掉padding的部分
      logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
      cross_entropy = cross_entropy * caption_mask[:,i]
      probs.append(logit_words)

      current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
      loss = loss + current_loss

    return loss, video, video_mask, caption, caption_mask, probs

#=====================================================================================
# Global Parameters
#=====================================================================================

video_train_feat_path = './MLDS_hw2_data/training_data/feat'
video_train_data_path = './MLDS_hw2_data/training_label.json'

model_path = './models'

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


def get_video_train_data(video_data_path, video_feat_path):
  video_data = pd.read_json(video_data_path)
  # # 只保留video裡'Language'欄位為English的資料
  # # 根據VideoID, Start, End欄位, 附檔名來新增一欄位：video_path
  # # axis為1時是套用到每個row(將每個row丟到lamda後面)
  video_data['video_path'] = video_data.apply(lambda row: row['id'] +'.npy', axis=1)
  # 結合檔名跟其所在的目錄成為新video_path
  video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
  # 確認路徑是否存在
  video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
  # 確認description欄位為String格式

  # 將所有的路徑取出變為一陣列, ex: a,b,c,c,d => [a, b, c, d]
  unique_filenames = sorted(video_data['video_path'].unique())
  # 去除重複的資料
  train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
  return train_data

def preProBuildWordVocab(sentence_iteratorArr, word_count_threshold=0):
  # borrowed this function from NeuralTalk
  word_counts = {}

  # nsent：總共有幾個"sentence"輸入 = len(captions)
  vocab = []
  nsents = 0
  for sentence_iterator in sentence_iteratorArr:
    for sent in sentence_iterator:
      nsents += 1
      # 將word取出
      for w in sent.lower().split(' '):
         # 將這個word與它的出現次數輸入進word_counts
        word_counts[w] = word_counts.get(w, 0) + 1
        # 若是word的出現次數沒有高於threhold的話就不新增進vocabulary
        if(word_counts[w] >= word_count_threshold and w not in vocab):
          vocab.append(w)
  print ('[ Word threshold: %d ] %d -> %d' % (word_count_threshold, len(word_counts), len(vocab)))

  ixtoword = {}
  ixtoword[0] = '<pad>'
  ixtoword[1] = '<bos>'
  ixtoword[2] = '<eos>'
  ixtoword[3] = '<unk>'

  wordtoix = {}
  wordtoix['<pad>'] = 0
  wordtoix['<bos>'] = 1
  wordtoix['<eos>'] = 2
  wordtoix['<unk>'] = 3

  # 將相對應的詞與index輸入進array
  for idx, w in enumerate(vocab):
    wordtoix[w] = idx + 4
    ixtoword[idx + 4] = w

  # 將這些特殊字元的出現次數變得跟句子數量一樣多
  word_counts['<pad>'] = nsents
  word_counts['<bos>'] = nsents
  word_counts['<eos>'] = nsents
  word_counts['<unk>'] = nsents

  # 根據各個字出現的次數來新增bias vector
  bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
  bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
  bias_init_vector = np.log(bias_init_vector)
  bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

  return wordtoix, ixtoword, bias_init_vector

def train():
  # 取得訓練及測試的資料, x為影片, y為字幕
  train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
  train_captions = train_data['caption'].values

  # 將字幕y_train跟y_test合在一起
  captions_list = list(train_captions)
  # ['哈','囉','呀'] => array(['哈','囉','呀'], dtype=object)
  captions = np.asarray(captions_list, dtype=np.object)
  # 取代字幕裡所有奇怪的符號
  regex = re.compile('(\.|\,|\"|\n|\\|/|)|(|]|&|\?|\!|\[)')
  new_capsArr = []

  for captionList in captions:
    captionList = map(lambda x: regex.sub('', x), captionList)
    new_capsArr.append(captionList)

  wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(new_capsArr, word_count_threshold=2)
  
  np.save("./mapping/wordtoix", wordtoix)
  np.save('./mapping/ixtoword', ixtoword)
  np.save("./mapping/bias_init_vector", bias_init_vector)

  model = Video_Caption_Generator(
      dim_image=dim_image,
      n_words=len(wordtoix),
      dim_hidden=dim_hidden,
      batch_size=batch_size,
      n_lstm_steps=n_frame_step,
      n_video_lstm_step=n_video_lstm_step,
      n_caption_lstm_step=n_caption_lstm_step,
      bias_init_vector=bias_init_vector)

  tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()

  train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
  tf.global_variables_initializer().run()
  # saver.restore(sess, './models/model-100')

  loss_record = open('loss.txt', 'w')

  for epoch in range(1, n_epochs + 1):

    index = list(train_data.index)
    np.random.shuffle(index)
    # .ix為pandas object的用法
    train_data = train_data.ix[index]

    current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc(np.random.choice(len(x)))[0])
    current_train_data = current_train_data.reset_index(drop=True)

    for start, end in zip(
        range(0, len(current_train_data), batch_size),
        range(batch_size, len(current_train_data), batch_size)):

      start_time = time.time()

      current_batch = current_train_data[start:end]

      current_videos = current_batch['video_path'].values

      current_feats = list(np.zeros((batch_size, n_video_lstm_step, dim_image)))
      current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
      current_video_masks = list(np.zeros((batch_size, n_video_lstm_step)))

      for ind,feat in enumerate(current_feats_vals):
        current_feats[ind][:len(current_feats_vals[ind])] = feat
        current_video_masks[ind][:len(current_feats_vals[ind])] = 1

      new_curr_captions = []
      current_captionsArr = current_batch['caption'].values

      for current_captionsList in current_captionsArr:
        current_captionsList = map(lambda x: '<bos> ' + x, current_captionsList)
        current_captionsList = map(lambda x: regex.sub('', x), current_captionsList)
        current_captionsList = list(current_captionsList)
        random_capIdx = randint(0, len(current_captionsList) - 1)
        new_current_caption = current_captionsList[random_capIdx]
        new_curr_captions.append(new_current_caption)

      for idx, each_cap in enumerate(new_curr_captions):
        word = each_cap.lower().split(' ')
        if len(word) < n_caption_lstm_step:
          new_curr_captions[idx] = new_curr_captions[idx] + ' <eos>'
        else:
          new_word = ''
          for i in range(n_caption_lstm_step-1):
            new_word = new_word + word[i] + ' '
          new_curr_captions[idx] = new_word + '<eos>'

      current_caption_ind = []
      for cap in new_curr_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
          if word in wordtoix:
            current_word_ind.append(wordtoix[word])
          else:
            current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)

      current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)

      current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
      current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
      nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) ))

      for ind, row in enumerate(current_caption_masks):
        last = nonzeros[ind] - 1
        row[:last] = 1

      _, loss_val, probs = sess.run(
          [train_op, tf_loss, tf_probs],
          feed_dict={
            tf_video: current_feats,
            tf_video_mask : current_video_masks,
            tf_caption: current_caption_matrix,
            tf_caption_mask: current_caption_masks,
          })


      tmp = randint(1, 190)
      for x in range(tmp, tmp + 10):
        ground_truth = ''
        for i in range(0, n_caption_lstm_step):
          a = [ixtoword[a] for a in list(current_caption_matrix[:, i])]
          ground_truth += a[x] + ' '
        print('[Ground Truth]\t', ground_truth)
        print('\n')
        prediction = ''
        for timestep in probs:
          prediction += ixtoword[np.argmax(timestep[x])] + ' '
        print('[Prediction]\t', '<bos> ' + prediction)
        print('====================================================================')


      print ('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
      loss_record.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

    if np.mod(epoch, 20) == 0:
      print ("Epoch ", epoch, " is done. Saving the model ...")
      saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

  loss_record.close()

train()
