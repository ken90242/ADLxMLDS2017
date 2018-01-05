import pandas as pd
import re
from Utils import skimage
import pickle
import skimage.io
import skimage.transform
import skipthoughts
import numpy as np

# original:25774\tblonde hair:25457\tdoll:1040\tdress:16585\tpink eyes:3896\ttagme:13190\t
TOTAL_NUM = 33430
class input_data:

	def __init__(self):
		self.pre_batch_position = 0
		self.test = {}

	def getLabels(self):
		pattern = re.compile('(\w[\w|\s]+\w)\s*\:\d+\t')
		picIdTags = {}

		src = pd.read_csv('tags_clean.csv', header=None)
		model = skipthoughts.load_model()
		for pid, raw in zip(src[0], src[1]):
			tags = re.findall(pattern, raw)
			filtered_tags = [t for t in tags if ('eyes' in t or 'hair' in t)] # 過濾與hair, eyes無關的詞彙
			if(len(filtered_tags) != 0): # 刪掉完全不符的資料
				print('Train skipthoughts: (idx):{}/(cap):{}'.format(pid, filtered_tags))
				# picIdTags[int(pid)] = filtered_tags
				picIdTags[int(pid)] = skipthoughts.encode(model, [' and '.join(filtered_tags)])[0]
				# self.test[' '.join(filtered_tags)] = skipthoughts.encode(model, [' '.join(fil/tered_tags)])

		# encoded_captions = {}
		# for i, img in enumerate(image_captions):
		# 	st = time.time()
		# 	encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
		# h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
		# for key in encoded_captions:
		# 	h.create_dataset(key, data=encoded_captions[key])
		# h.close()
		return picIdTags

	def getFeatures(self, path='faces/'):
		picIdFeas = {}
		for pid in range(TOTAL_NUM + 1):
			img = skimage.io.imread('{}{}.jpg'.format('faces/', pid))
			# img_resized = skimage.transform.resize(img, (64, 64))
			img_resized = img
			picIdFeas[pid] = img_resized
		return picIdFeas

	def read_data_sets(self):
		# if(True):
		# 	f, l = pickle.load(open('dataset.pkl', 'rb'))
		# 	self.p_features = f
		# 	self.p_labels = l
		# 	return None

		o_labels = self.getLabels() # original
		o_features = self.getFeatures()
		p_labels = [] # processed
		p_features = []

		for i in o_labels:
			try: # 會有i不存在的可能(因為會先刪除跟hair, eyes無關的詞彙)
				p_labels.append(o_labels[i])
				p_features.append(o_features[i])
			except:
				print('[Error] index {} doesn\'t exist'.format(i))

		self.p_labels = p_labels
		self.p_features = p_features

		pickle.dump((p_features, p_labels), open('../Data/dataset.pkl', 'wb'))

	def next_batch(self, batch_size):
		if (self.pre_batch_position >= 18111):
			self.pre_batch_position = 0
		batch_labels = self.p_labels[self.pre_batch_position : self.pre_batch_position + batch_size]
		batch_labels = [x[:4800] for x in batch_labels]
		batch_features = self.p_features[self.pre_batch_position : self.pre_batch_position + batch_size]
		# batch_features = [x[:6] for x in batch_features]
		self.pre_batch_position += batch_size
		self.pre_batch_position %= TOTAL_NUM
		return (batch_features, batch_labels)

if __name__ == '__main__':
	a = input_data()

	print(d)
