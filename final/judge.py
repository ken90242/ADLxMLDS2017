import judger_medical as judger
import tensorflow as tf
from importlib import import_module

vgg19 = import_module('vgg19')
test = import_module('test')



imgs = judger.get_file_names()
f = judger.get_output_file_object()
sess = tf.Session()
vgg = vgg19.Vgg19('test-save_24.npy')
vgg.build()
sess.run(tf.global_variables_initializer())

for img in imgs:
    img_data = test.load_image(img)

    answer = test.inference(img_data,sess,vgg)
    print (img, answer.number_of_boxes)
    f.write('%s %d\n' % (img, answer.number_of_boxes))
    for box in answer.boxes:
        print (box.class_name, box.x, box.y, box.w, box.h)
        f.write('%s %f %f %f %f\n' % (box.class_name, box.x, box.y, box.w, box.h))
score, err = judger.judge()

