import pickle
from tqdm import tqdm
import sys
DATA_PATH = sys.argv[1]

# build pkl for transforming: idx2har.pkl, phone2idx.pkl
# build correct label sequence: aligned_train.lab

print ('[INITIALIZE] start to setup...')

prephone2postphone = {}
prephone2char = {}
idx2prephone = {}
prephone2idx = {}

idx2postphone = {}
prephone2vec = {}
idx2char = {}

idx2char39 = {}
postphone2vec39 = {}
phone2vec39 = {}

with open(DATA_PATH + '/phones/48_39.map') as f:
	for line in tqdm(f, desc='phone2vec'):
		(prephone, postphone) = line.split('\t')
		prephone2postphone[prephone.replace('\n', '')] = postphone.replace('\n', '')

# 	res = {}
# 	post_phones = []
# 	pre_phones = []
# 	for line in tqdm(f, desc='phone2vec'):
# 		(pre, post) = line.split('\t')
# 		post_phones.append(post.replace('\n', ''))
# 		pre_phones.append(pre.replace('\n', ''))
# 		prephone2postphone[pre.replace('\n', '')] = post.replace('\n', '')
# 	for i, postphone in enumerate(list(set(post_phones))):
# 		vec = [0] * 39
# 		vec[i] = 1
# 		postphone2vec39[postphone] = vec
# 		idx2postphone[i] = postphone
# 	for i, prephone in enumerate(pre_phones):
# 		vec = [0] * 48
# 		vec[i] = 1
# 		prephone2vec[prephone] = vec
# 		idx2prephone[i] = prephone
# 	for prephone in prephone2vec:
# 		postphone = prephone2postphone[prephone]
# 		phone2vec39[prephone] = postphone2vec39[postphone]

with open(DATA_PATH + '/48phone_char.map') as f:
	for line in f:
		(prephone, number, char) = line.split()
		prephone2char[prephone] = char
		idx = int(number)
		vec = [0] * 48
		vec[idx] = 1
		prephone2vec[prephone] = vec
		idx2prephone[idx] = prephone
		prephone2idx[prephone] = int(idx)


for idx, prephone in tqdm(idx2prephone.items(), desc='idx2char'):
	postphone = prephone2postphone[prephone]
	char = prephone2char[postphone]
	idx2char[idx] = char
# for idx, postphone in tqdm(idx2postphone.items(), desc='idx2char39'):
# 	char = postphone2char[postphone]
# 	idx2char39[idx] = char

pickle.dump(idx2char, open('idx2char.pkl', 'wb'))
# pickle.dump(idx2char39, open('idx2char39.pkl', 'wb'))
pickle.dump(prephone2idx, open('prephone2idx.pkl', 'wb'))
pickle.dump(prephone2vec, open('phone2vec.pkl', 'wb'))
# pickle.dump(phone2vec39, open('phone2vec39.pkl', 'wb'))


correct_seq = []

with open(DATA_PATH + '/fbank/train.ark') as f:
	for line in tqdm(f, desc='(tmp)sequence_array'):
		key = line.split()[0]
		correct_seq.append(key)
f.closed

raw_label_dict = {}
with open(DATA_PATH + '/label/train.lab') as f:
	for line in tqdm(f, desc='(tmp)raw_dict'):
			(key, val) = line.split(',')
			raw_label_dict[key] = val
f.closed

with open('aligned_train.lab', 'w') as f:
	for seq in tqdm(correct_seq, desc='aligned_train.lab'):
		f.write(seq + ',' + raw_label_dict[seq])
f.closed

print ('[FINISHED] complete.')