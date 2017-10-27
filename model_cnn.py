import numpy as np
from numpy import asarray, zeros

import keras
from keras import initializers, backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Masking, Bidirectional, BatchNormalization, Conv1D
from keras.optimizers import adam

from tqdm import tqdm
import mmap
import pickle

keras.initializers.Orthogonal(gain=1.0, seed=None)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')

phone2vec = pickle.load(open('phone2vec.pkl', 'rb'))


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines



TRAIN_DIMENSION = 69
WINDOW_SIZE = 31

y_train = []
with open('aligned_train.lab') as f:
	for line in tqdm(f, desc='aligned_train.lab', total=get_num_lines('aligned_train.lab')):
		(ids, phone) = line.split(',')
		phone = phone.replace('\n', '')
		y_train.append(phone2vec[phone])
f.closed

print(np.array(y_train).shape)



a = []
prev_id = ''
f = open('../data/fbank/train.ark')
arr = []
for line in tqdm(f, desc='fbank/train.ark', total=get_num_lines('../data/fbank/train.ark')):
	values = line.split()
	(q, w, e) = values[0].split('_')
	curr_id = q + '_' + w
	coefs = asarray(values[1:], dtype='float32')

	if(prev_id == ''):
		prev_id = curr_id
	
	if(prev_id == curr_id):
		arr.append(coefs)
	else:
		a.append(arr)
		prev_id = curr_id
		arr = [coefs]
f.close()
a.append(arr)

x_train = []
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
		x_train.append(window)
input_shape = np.array(x_train).shape
print(input_shape)
model = Sequential()
# model.add(Masking(mask_value=0.))
model.add(Conv1D(128, 
                 padding = 'causal', 
                 kernel_size = 8,
                 input_shape=(input_shape[1], input_shape[2])))
model.add(BatchNormalization())
model.add(LSTM(512, return_sequences=False))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(48, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.1, batch_size=500, callbacks=[checkpoint, earlystopping])
model.save('result_model.h5')
backend.clear_session()
