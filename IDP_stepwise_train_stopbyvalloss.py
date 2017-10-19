# coding: utf-8

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import cPickle
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from model import model

os.environ['CUDA_VISIBLE_DEVICES'] = ''

profile   = sys.argv[1]
alpha     = float(sys.argv[2])
alternate = int(sys.argv[3])
act_type  = sys.argv[4]

early_stop = "val_accu"

''' MAIN PROGRAM '''

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

W1 = np.random.uniform(high=0.1,low=-0.1,size=(784,100)).astype('float32')
b1 = np.random.uniform(high=0.1,low=-0.1,size=(100)).astype('float32')
W2 = np.random.uniform(high=0.1,low=-0.1,size=(100,100)).astype('float32')
b2 = np.random.uniform(high=0.1,low=-0.1,size=(100)).astype('float32')   
W3 = np.random.uniform(high=0.1,low=-0.1,size=(100,10)).astype('float32')
b3 = np.random.uniform(high=0.1,low=-0.1,size=(10)).astype('float32')

params = {'W1':W1,'W2':W2,'W3':W3,'b1':b1,'b2':b2,'b3':b3}

counter = 0
trained_idp = np.arange(1.0,0.05,-0.1)
# trained_idp = np.array([0.3,0.7])
tested_idp = np.arange(0.1,1.05,0.05)

### training using gradually complete loss function
counter = counter + 1
params['idp'] = trained_idp
params['alpha'] = alpha
params['act_type'] = act_type

print(params['idp'])
print("============ ITERATIVE TRAINING ============")
with tf.Session() as sess:
    with tf.variable_scope(profile+str(counter)) as scope:
        intv = 0
        tidp = 0
        var_reuse = False
        go_on = True
        current_best_loss = 0
        current_best_counter = 0

        """ result containers """
        profile_arr = []
        idp_arr = []
        accu_arr = []
        r2_dict = {}
        log = pd.DataFrame()
        
        while go_on:
            save_dir = "~/IDP/output_stepwise/"
            epochs = 1
            batch_per_epoch = 200
            batch_size = 28
            learning_rate = 0.001
            config = {'epochs': epochs,
                      'batch_per_epoch': batch_per_epoch,
                      'batch_size': batch_size,
                      'learning_rate': learning_rate,
                      'save_dir': save_dir,
                      'log_file': "{0}{1}_{2}_{3}_log.csv".format(save_dir,profile,int(alpha*100),alternate),
                      'result_file': "{0}{1}_{2}_{3}_result.csv".format(save_dir,profile,int(alpha*100),alternate),
                      'r2_file': "{0}{1}_{2}_{3}_r2.csv".format(save_dir,profile,int(alpha*100),alternate),
                      'mnist': mnist,
                     }
        
            intv = intv + 1
            print("Start "+str(intv)+" round")

            ''' training W,b '''
            if intv == 1:
                model_test = model(profile,params=params)
                model_test.set_trained_idp(sess=sess,loss_counter = tidp)
                model_test.loss_counter = tidp
                print("=====< Start to optimize IDP = %d >=====" % (trained_idp[tidp]*100))
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=False,reuse=var_reuse,verbose=True)
                this_loss = log1[early_stop][0]
                log = pd.DataFrame.from_dict(log1)
                r2_dict['initial'] = sess.run(model_test.r2)
            else:
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=False,reuse=var_reuse,verbose=True)
                this_loss = log1[early_stop][0]
                log1 = pd.DataFrame.from_dict(log1)
                log = pd.concat([log,log1])
            
            if alternate:
                ''' test after W,b updates '''
                true_lab = [np.argmax(ite) for ite in mnist.test.labels]
                for idp in tested_idp:
                    probs = model_test.predict(sess=sess,config=config,idp=idp,scope=profile+str(counter)+'i'+str(int(idp*100)),reuse=var_reuse)
                    pred_lab = [np.argmax(ite) for ite in probs[0]]
                    accu = accuracy_score(y_pred=pred_lab,y_true=true_lab)
                    # print("IDP={:.2f}, accuracy={:.3f}".format(idp,accu))
                    profile_arr.append("{0}(train W,b in the {1} epochs at IDP = {2})".format(profile,intv,model_test.idp[model_test.loss_counter]))
                    idp_arr.append(idp)
                    accu_arr.append(accu)
                var_reuse = True

            ''' training gamma '''
            #scope.reuse_variables()
            if alternate:
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=True,reuse=var_reuse,verbose=True)
                this_loss = log1[early_stop][0]
                log1 = pd.DataFrame.from_dict(log1)
                log = pd.concat([log,log1])
                r2_dict['after_'+str(intv)] = sess.run(model_test.r2)

            ''' test after gamma updates '''
            true_lab = [np.argmax(ite) for ite in mnist.test.labels]
            this_accu = []
            for idp in tested_idp:  
                probs = model_test.predict(sess=sess,config=config,idp=idp,scope=profile+str(counter)+'i'+str(int(idp*100)),reuse=var_reuse)
                pred_lab = [np.argmax(ite) for ite in probs[0]]
                accu = accuracy_score(y_pred=pred_lab,y_true=true_lab)
                # print("IDP={:.2f}, accuracy={:.3f}".format(idp,accu))
                profile_arr.append("{0}(train gamma in the {1} epochs at IDP = {2})".format(profile,intv,model_test.idp[model_test.loss_counter]))
                idp_arr.append(idp)
                accu_arr.append(accu)
                this_accu.append(accu)

            var_reuse = True # temporary
            
            if this_loss > current_best_loss:
                current_best_loss = this_loss
                current_best_counter = 0
            else:
                current_best_counter = current_best_counter + 1
            
            print("current focused performance: %f, current best counter: %d" % (this_loss,current_best_counter))
            
            # if converged and wait for 2*2 epochs, next idp point
            if (current_best_counter >= 2): #or (this_accu[tidp] >= np.min([0.9+tidp*0.03,0.999])):
                tidp = tidp + 1
                model_test.loss_counter = tidp

                if tidp == len(params['idp']):
                    go_on = False
                    break
                current_best_counter = 0
                current_best_loss = 0
                
                # set trained idp
                model_test.set_trained_idp(sess=sess, loss_counter = tidp)
                print("=====< Start to optimize IDP = %d >=====" % (trained_idp[tidp]*100))
                 
        r2 = pd.DataFrame.from_dict(r2_dict)
        r2.to_csv(config['r2_file'],index=None)
        log.to_csv(config['log_file'],index=None)
        res = pd.DataFrame.from_dict({'profile':profile_arr,'IDP':idp_arr,'accu':accu_arr})
        res.to_csv(config['result_file'],index=None)
        print(res)
