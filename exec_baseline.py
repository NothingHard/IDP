import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import cPickle
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from baseline_model import baseline_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cand = range(10,100+1,5)

for hidden in cand:
    params = {}
    """ setting from config """
    save_dir = "baseline/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    """ save_dir as filename """
    save_dir = save_dir+str(hidden)
    epochs = 100
    batch_per_epoch = 200
    batch_size = 32
    learning_rate = 0.001

    """ read input dataset """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    """ record container """
    val_loss_per_epoch = []
    val_accu_per_epoch = []
    loss_per_epoch = []
    accu_per_epoch = []
    log = pd.DataFrame()

    print("============ START TRAINING ============")
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    current_best_loss = 1000
    current_best_counter = 0

    with tf.Session(config=gpu_config) as sess:
        with tf.variable_scope("s"+str(hidden)) as scope:
            
            """ create a baseline model"""
            params['learning_rate'] = learning_rate
            params['hidden'] = hidden
            model = baseline_model(params)
            
            sess.run(tf.global_variables_initializer())
            
            """ model saver"""
            saver = tf.train.Saver()

            """ training """
            for epoch in range(0,epochs):
                start_time = time.time()
                train_loss = []
                train_accu = []
                for batch in range(0,batch_per_epoch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                    _,loss, accu = sess.run([model.train_op, model.loss, model.accuracy], 
                                                    feed_dict = {model.x:batch_x, model.y: batch_y})
                    train_loss.append(loss)
                    train_accu.append(accu)

                loss_per_epoch.append(np.mean(train_loss))
                accu_per_epoch.append(np.mean(train_accu))

                """ validation """
                val_loss , val_accu = sess.run([model.loss,model.accuracy],
                                                feed_dict={model.x:mnist.test.images, model.y:mnist.test.labels})
                val_loss_per_epoch.append(val_loss)
                val_accu_per_epoch.append(val_accu) 


                print("epoch: %2d \t time: %2f \t loss: %2f \t val_loss: %2f \t accu: %2f \t val_accu: %2f"% 
                  (epoch,time.time() - start_time,np.mean(train_loss), val_loss, np.mean(train_accu), val_accu))

                if val_loss < current_best_loss:
                    current_best_loss = val_loss
                    current_best_counter = 0
                else:
                    current_best_counter = current_best_counter + 1

                print("current focused performance: %f, current best counter: %d" % (val_loss,current_best_counter))

                # if converged and wait for 2*2 epochs, next idp point
                if (current_best_counter >= 4):
                    print("Early stopping!")
                    save_path = saver.save(sess, save_dir+"model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    break

            res = {'loss':loss_per_epoch, 'val_loss':val_loss_per_epoch,'accu':accu_per_epoch, 'val_accu':val_accu_per_epoch}
            res = pd.DataFrame.from_dict(res)
            res.to_csv(save_dir+'result.csv',index=None)