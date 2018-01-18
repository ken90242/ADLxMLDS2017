import numpy as np
import tensorflow as tf
import random
import vgg19
import os
import pandas as pd


threshold_value=0.25
batch_size=64
maxnum=8000
epochs=150
learning_rate=1e-4
modeldir='./model'

def main():

    '''    
    image_idxs=[]
    # load the training image and label
    file=open('train.txt','r')

    line=file.readline()

    while line:
        image_idx=line.strip()
        image_idxs.append(image_idx)
        line=file.readline()
    '''
    
    #load the traing & validation image and label

    total_validation_list = load_idlist('valid.txt')
    total_training_list = load_idlist('pool.txt')
    sess = tf.Session()
    

    model = vgg19.Vgg19('vgg19.npy')
    model.build(learning_rate)

    sess.run(tf.global_variables_initializer())
    total_validation_image, total_validation_label = load_trainingdata(total_validation_list, 440)

    total_train_image, total_train_label = load_trainingdata(total_training_list, 6589)

    trainloadlist = np.arange(0, len(total_train_label)-1)
    validloadlist = np.arange(0, len(total_validation_label)-1)
    threshold=np.zeros((batch_size,len(total_validation_label[0])))
    threshold.fill(threshold_value)
    threshold=np.asarray(threshold)

    progress=pd.DataFrame(["epochs","batch_no","train loss","val_loss","acc","pos pred"])

    # traing stage
    i = 0
    weight=51301/8000
    while True:

        batch_no = 0
        np.random.shuffle(trainloadlist)
        np.random.shuffle(validloadlist)

        while (batch_no+1)*batch_size < len(total_train_image):

            train_image, train_label = get_training_batch(batch_size, total_train_image, total_train_label, trainloadlist,batch_no)

            validation_image,validation_label=get_valid_batch(batch_size,total_validation_image,total_validation_label)

            sess.run(model.train, feed_dict={model.input: train_image, model.GT: train_label, model.threshold:threshold,
                                             model.weight:weight, model.train_mode: True})

         
            if batch_no%30==0:
                train_loss=sess.run(model.loss, feed_dict={model.input: train_image, model.GT: train_label, model.threshold:threshold,
                                                           model.weight: weight, model.train_mode: False})
                val_loss=sess.run(model.loss, feed_dict={model.input: validation_image, model.GT: validation_label,
                                                         model.threshold:threshold,model.weight:weight, model.train_mode: False})
                acc = sess.run(model.accuracy, feed_dict={model.input: validation_image, model.GT: validation_label,model.threshold:threshold,
                                                           model.train_mode: False})
                pos = sess.run(model.positive, feed_dict={model.input: validation_image, model.GT: validation_label,model.threshold:threshold,
                                                          model.train_mode: False})

                print ("epochs:",i,"batch_no:",batch_no,"train loss:",train_loss,"val_loss:",val_loss,"acc:",acc,"pos pred:", pos )
                progress=progress.append(pd.Series([i,batch_no,train_loss,val_loss,acc, pos]),ignore_index=True)

                pred = sess.run(model.prob, feed_dict={model.input: validation_image, model.GT: validation_label,model.threshold:threshold,
                                                       model.train_mode: False})
                prediction = sess.run(model.prediction, feed_dict={model.input: validation_image, model.GT: validation_label,
                                                       model.threshold: threshold,
                                                       model.train_mode: False})
                predictionlist = []
                for number in range(len(pred)):
                    predictionlist.append(pred[number])
                    predictionlist.append(prediction[number])
                    predictionlist.append(validation_label[number])
                predictionlist = pd.DataFrame(predictionlist)
                if not os.path.exists(modeldir):
                    os.makedirs(modeldir)
                predictionlist.to_csv(os.path.join(modeldir,'check_'+str(batch_no)+'.csv'), index=None, header=None)


            batch_no+=1
        if i%5==0:
          if not os.path.exists(modeldir):
             os.makedirs(modeldir)
          model.save_npy(sess, os.path.join(modeldir,'test-save_{}.npy'.format(i)))
          progress.to_csv(os.path.join(modeldir,'record_1.csv'),header=None)
        i+=1

def load_idlist(filename):
    idlist=[]
    file=open(filename,'r')

    line=file.readline()

    while line:
        image_idx=line.strip()
        idlist.append(image_idx)

        line = file.readline()
    return idlist

def load_trainingdata(idlist,maxnum):

    imagelist=[]
    labellist=[]
    count=0
    while count < maxnum:
         image_idx = idlist[count]
         if os.path.exists('data/' + image_idx + '.npy'):
             image = np.load('data/' + image_idx + '.npy')
             label = np.load('labels/' + image_idx + '.npy')
             imagelist.append(image)
             labellist.append(label[1:9])
             count+=1
    return imagelist,labellist

def get_training_batch(batch_size,image,label, trainloadlist,batch_no):
    train_image=[]
    train_label=[]

    for i in range(batch_no*batch_size,batch_no*batch_size+batch_size):
        idx=trainloadlist[i]
        train_image.append(image[idx])
        train_label.append(label[idx])


    return train_image,train_label


def get_valid_batch(batch_size,image,label):
    train_image=[]
    train_label=[]

    for i in range(batch_size):
        idx=random.randint(0, len(image)-1)
        train_image.append(image[idx])
        train_label.append(label[idx])

    return train_image,train_label

if __name__ == '__main__':
    main()

