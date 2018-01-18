import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import pandas as pd
import os
import cv2
from importlib import import_module

vgg19 = import_module('vgg19')


class_names = pd.read_csv('label.txt', header=None)
class_names.columns = ['index', 'class']


def load_image(img_path):
    print("Loading image")
    img = imread(img_path, mode='RGB')
    img = imresize(img, (224, 224))
    # Converting shape from [224,224,3] tp [1,224,224,3]
    x = np.expand_dims(img, axis=0)
    # Converting RGB to BGR for VGG
    x = x[:, :, :, ::-1]
    return x 


def grad_cam(x, model, sess, predicted_class, nb_classes):
    print("Setting gradients to 1 for target class and rest to 0")
    # Conv layer tensor [?,7,7,512]
    conv_layer = model.pool5
    # [1000]-D tensor with target class index set to 1 and rest as 0
    one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
    signal = tf.multiply(model.fc8, one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={model.input: x,model.train_mode:False})
    output = output[0]  # [7,7,512]
    grads_val = grads_val[0]  # [7,7,512]

    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = imresize(cam, (224, 224))

    # Converting grayscale to 3-D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3, [1, 1, 3])

    return cam3


class image_block():
    def __init__(self):
        self.boxes=[]
        self.number_of_boxes=len(self.boxes)


class box_block():
    def __init__(self,class_name,x,y,w,h):
        self.class_name=class_name
        self.x=x
        self.y=y
        self.w=w
        self.h=h


def inference(input_image,sess,vgg):

    prob = sess.run(vgg.prob, feed_dict={vgg.input: input_image, vgg.train_mode:False})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    possible_cause = []
    for p in preds:
        if prob[p]>0.25:
            possible_cause.append([p,class_names['class'][p], prob[p]])

    nb_classes = 8


    cause_image=[]
    for i in range (len(possible_cause)):
        predicted_class = possible_cause[i][0]
        cam3 = grad_cam(input_image, vgg, sess, predicted_class, nb_classes)
        cause_image.append(cam3)

    answer=image_block()  
    # Superimposing the visualization with the image.
    for i in range(len(cause_image)):
       #cv2.imwrite(str(i)+'_'+str(possible_cause[i][1])+'.png', cause_image[i])
       image,_,_=cv2.split(cause_image[i])
       _, maxval, _, maxLoc = cv2.minMaxLoc(image)
          
       x=maxLoc[0]
       y=maxLoc[1]
       half_width=0
       half_height=0
       while image[y][x]>0.9 and x<223:
           x+=1
           half_width+=1

       x=maxLoc[0]     
     
       while image[y][x]>0.9 and y<223:
           y+=1
           half_height+=1           
       y=maxLoc[1]
           
       if x-half_width<0:
           upx=0
           width=half_width+x
       else:
           upx=x-half_width
           width=2*half_width

       if y-half_height<0:
           upy=0
           height=half_height+y
       else:
           upy=y-half_height
           height=2*half_height
       scale=1024.0/224.0
       if possible_cause[i][1]!="No Finding":
           box=box_block(possible_cause[i][1],upx*scale,upy*scale,width*scale,height*scale)

           answer.boxes.append(box)       

       answer.number_of_boxes=len(answer.boxes)
    return answer
    # reference github https://github.com/Ankush96/grad-cam.tensorflow.git
