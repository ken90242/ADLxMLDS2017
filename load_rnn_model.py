import sys
DATA_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

import numpy as np
from numpy import asarray
from keras.models import load_model
from keras.layers import Dense, Dropout
import itertools
from keras.utils import np_utils
import keras

from collections import OrderedDict

from tqdm import tqdm
import mmap
import pickle
import time
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

idx2char = pickle.load(open('./idx2char.pkl', 'rb'))

model = load_model('./rnn_model.h5')
print(model.summary())


TRAIN_DIMENSION = 69
WINDOW_SIZE = 31


a = []
ids_arr = []
ids_length = {}
prev_id = ''

x_train = []
with open(DATA_PATH + '/fbank/train.ark') as f:
	for line in f:
		values = line.split()
		coefs = asarray(values[1:], dtype='float32')
		x_train.append(coefs)
sc.fit(x_train)

x_test_sentenceId = []
x_test_input = []
with open(DATA_PATH + '/fbank/test.ark') as f:
	for line in f:
		values = line.split()
		sentenceId = values[0]
		coefs = asarray(values[1:], dtype='float32')
		x_test_sentenceId.append(sentenceId)
		x_test_input.append(coefs)

x_test_std = sc.transform(x_test_input)




arr = []
f = zip(x_test_sentenceId,x_test_input)

for sentenceId, coefs in tqdm(f):
	(q, w, e) = sentenceId.split('_')
	curr_id = q + '_' + w

	if(prev_id == ''):
		prev_id = curr_id
	
	if(prev_id == curr_id):
		arr.append(coefs)
	else:
		a.append(arr)
		ids_arr.append(prev_id)
		ids_length[prev_id] = len(arr)
		prev_id = curr_id
		arr = [coefs]
a.append(arr)
ids_arr.append(prev_id)
ids_length[prev_id] = len(arr)

x_test = []
for sentence in tqdm(a, total=len(a)):
	for idx, x in enumerate(sentence):
		window = []
		start = -int((WINDOW_SIZE-1)/2)
		end = int((WINDOW_SIZE+1)/2)
		for i in range(start, end):
			if(idx+i<0 or idx+i > len(sentence)-1):
				pad = [0.] * TRAIN_DIMENSION
				window.append(pad)
			else:
				window.append(sentence[idx+i])
		x_test.append(window)
input_shape = np.array(x_test).shape
print(input_shape)

chars = []
prediction_items = model.predict(x_test, verbose=1)
for item in tqdm(prediction_items, total=len(prediction_items)):
	prediction = idx2char[np.argmax(item)]
	chars.append(prediction)
	

result = OrderedDict()
start = 0
end = 0
for sentence_id in ids_arr:
	start = end
	end = start + ids_length[sentence_id]
	string = ''.join(chars[start:end])
	string = ''.join(char for char, _ in itertools.groupby(string))
	if(len(string) != 0):
		if(string[0] == 'L'):
			string = string[1:]
		if(string[-1] == 'L'):
			string = string[:-1]
	result[sentence_id] = string



with open(OUTPUT_PATH, 'w') as f:
	f.write('id,phone_sequence\n')
	for ids in result:
		f.write(ids+','+result[ids]+'\n')
f.closed
