import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import os
import sys

# read the label index 
label_index={}
f1=open('label.txt','r')
line=f1.readline()
while line:
    tmp=line.split(',')
    tmp[1]=tmp[1].strip()
    label_index[tmp[1]]=tmp[0]
    line=f1.readline()

f1.close()




file=open('Data_Entry_2017_v2.csv','r')

line=file.readline()

k=0
for line in file.readlines()[70050:len(file.readlines())-1]:
    print (line)
    k += 1
    tmp=line.split(',')
    tmp_label=np.zeros(len(label_index))
    if '|' in tmp[1]:
        tmp1=tmp[1].split('|')
        for i in range(len(tmp1)):
         #   print (tmp1[i])
#            tmp_label[int(label_index[tmp1[i]])]=1.0/len(tmp1)
            tmp_label[int(label_index[tmp1[i]])] = 1.0

    else:
        tmp_label[int(label_index[tmp[1]])]=1        
    line=file.readline()
    # sys.argv[1] = images/ (default)
    if os.path.exists(os.path.join(sys.argv[1], tmp[0])):
      img=skimage.io.imread(os.path.join(sys.argv[1], tmp[0]),True)
      img_stack=np.stack((img,img,img),axis=0)
      img_stack=np.transpose(img_stack,(1,2,0))
      img_resized=skimage.transform.resize(img_stack,(224,224))
      #print (img_resized.shape)
    
      np.save('data/'+tmp[0],img_resized)

      np.save('labels/'+tmp[0],tmp_label)
       
    
