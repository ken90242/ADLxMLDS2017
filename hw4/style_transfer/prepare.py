import os
import re
import pandas as pd
import pickle as pkl
import numpy as np
import shutil
import random
from os.path import join, exists
from skimage.io import imread

TOTAL_NUM = 33430
class Prepare:

	def __init__(self):
		self.datasets = {}

	def getLabels(self):
		# A為一般，B為棕頭髮
		TrainA = []
		TrainB = []

		src = pd.read_csv('../tags_clean.csv', header=None)
		for pid, raw in zip(src[0], src[1]):
			is_RedHair = False
			if ('brown hair' in raw):
				is_RedHair = True

			if(is_RedHair == True):
				TrainB.append(pid)
			else:
				TrainA.append(pid)

		self.datasets['trainA'] = random.sample(TrainA, 5000)
		self.datasets['trainB'] = TrainB

		self.datasets['testA'] = random.sample(TrainA, 25)
		self.datasets['testB'] = random.sample(TrainB, 25)

		self.move()

	def move(self, srcDir='faces', descDir='datasets/hair2brown'):
		if(exists(descDir)): shutil.rmtree(descDir)
		os.mkdir(descDir)
		for dirType in ['trainA', 'trainB', 'testA', 'testB']:
			if not exists(join(descDir, dirType)): os.mkdir(join(descDir, dirType)) 
			desc = join(descDir, dirType)
			for imgId in self.datasets[dirType]:
				img = str(imgId) + '.jpg'
				src = join(srcDir, img)

				shutil.copy(src, desc)
		print('move done')
		self.build_train_pkl()

	def build_train_pkl(self):
		trainImgs = {}
		for imgId in self.datasets['trainA']:
			img = imread(join('datasets/hair2brown', 'trainA', str(imgId) + '.jpg'))
			trainImgs[str(imgId)] = img
		for imgId in self.datasets['trainB']:
			img = imread(join('datasets/hair2brown', 'trainB', str(imgId) + '.jpg'))
			trainImgs[str(imgId)] = img
		for imgId in self.datasets['testA']:
			img = imread(join('datasets/hair2brown', 'testA', str(imgId) + '.jpg'))
			trainImgs[str(imgId)] = img
		for imgId in self.datasets['testB']:
			img = imread(join('datasets/hair2brown', 'testB', str(imgId) + '.jpg'))
			trainImgs[str(imgId)] = img

		pkl.dump(trainImgs, open('trainImgs.pkl', 'wb'))

		print('pkl built done')

if __name__ == '__main__':
	Prepare().getLabels()
