
# coding: utf-8

# get_ipython().magic('env CUDA_DEVICE_ORDER=PCI_BUS_ID')
# get_ipython().magic('env CUDA_VISIBLE_DEVICES=4')
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

profile   = sys.argv[1]
alpha     = float(sys.argv[2])
alternate = int(sys.argv[3])
act_type  = sys.argv[4]

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
trained_idp = np.arange(0.1,1.05,0.1)
tested_idp = np.arange(0.1,1.05,0.05)

### spec definition here
# get bins from trained_idp
if len(trained_idp)>1:
    bins = [0] + ((trained_idp[0:-1]+trained_idp[1:])/2).tolist() + [1]
else:
    bins = [0] + trained_idp.tolist() + [1]

spts = np.append(np.random.normal(loc=0.3,scale=0.1,size=100000),
                    np.random.normal(loc=0.7,scale=0.1,size=100000))
from scipy.stats.kde import gaussian_kde
KDEpdf = gaussian_kde(spts)
spec = np.histogram(spts,bins)[0]
spec = spec / np.sum(spec)
print("spec in use:")
print(spec)

### training using gradually complete loss function
params['idp'] = trained_idp
params['alpha'] = alpha
params['act_type'] = act_type
all_epoch = 50

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
        save_dir = "~/IDP/output/"
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
        while go_on:
            all_epoch -= 1
            intv += 1
            print("Start "+str(intv)+" round")

            ''' training W,b '''
            if intv == 1:
                model_test = model(profile,params=params)
                model_test.set_random_idp(sess=sess,probs=spec)
                tidp = model_test.loss_counter
                print("=====< Start to optimize IDP = %d >=====" % (trained_idp[tidp]*100))
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=False,reuse=var_reuse,verbose=True)
                this_loss = log1['val_loss'][0]
                log = pd.DataFrame.from_dict(log1)
                r2_dict['initial'] = sess.run(model_test.r2)
            else:
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=False,reuse=var_reuse,verbose=True)
                this_loss = log1['val_loss'][0]
                log1 = pd.DataFrame.from_dict(log1)
                log = pd.concat([log,log1])
            

            ''' test after W,b updates '''
            true_lab = [np.argmax(ite) for ite in mnist.test.labels]
            this_accu = []
            this_weight = []
            for idp in tested_idp:
                probs = model_test.predict(sess=sess,config=config,idp=idp,scope=profile+str(counter)+'i'+str(int(idp*100)),reuse=var_reuse)
                pred_lab = [np.argmax(ite) for ite in probs[0]]
                accu = accuracy_score(y_pred=pred_lab,y_true=true_lab)
                # print("IDP={:.2f}, accuracy={:.3f}".format(idp,accu))
                profile_arr.append("{0}(train W,b in the {1} epochs at IDP = {2})".format(profile,intv,model_test.idp[model_test.loss_counter]))
                idp_arr.append(idp)
                accu_arr.append(accu)
                this_accu.append(accu)
                this_weight.append(KDEpdf(idp))
            var_reuse = True
            
            ''' training gamma '''
            #scope.reuse_variables()
            if alternate:
                intv += 1
                log1 = model_test.train(sess=sess,config=config,gamma_trainable=True,reuse=var_reuse,verbose=False)
                this_loss = log1['val_loss'][0]
                log1 = pd.DataFrame.from_dict(log1)
                log = pd.concat([log,log1])
                r2_dict['after_'+str(intv)] = sess.run(model_test.r2)

                ''' test after gamma updates '''
                true_lab = [np.argmax(ite) for ite in mnist.test.labels]
                this_accu = []
                this_weight = []
                for idp in tested_idp:  
                    probs = model_test.predict(sess=sess,config=config,idp=idp,scope=profile+str(counter)+'i'+str(int(idp*100)),reuse=var_reuse)
                    pred_lab = [np.argmax(ite) for ite in probs[0]]
                    accu = accuracy_score(y_pred=pred_lab,y_true=true_lab)
                    # print("IDP={:.2f}, accuracy={:.3f}".format(idp,accu))
                    profile_arr.append("{0}(train gamma in the {1} epochs at IDP = {2})".format(profile,intv,model_test.idp[model_test.loss_counter]))
                    idp_arr.append(idp)
                    accu_arr.append(accu)
                    this_accu.append(accu)
                    this_weight.append(KDEpdf(idp))

            var_reuse = True # temporary
            this_weight = np.array(this_weight)
            this_accu = np.array(this_accu)
            this_accu = this_accu*this_weight/np.sum(this_weight)

            if this_loss < current_best_loss:
                current_best_loss = this_loss
                current_best_counter = 0
            else:
                current_best_counter = current_best_counter + 1
            
            print("current focused performance: %f, current best counter: %d" % (np.sum(this_accu),current_best_counter))
            
            # if converged and wait for 2*2 epochs, next idp point
            if (current_best_counter >= 2): #or (this_accu[tidp] >= np.min([0.9+tidp*0.03,0.999])):
                current_best_counter = 0
                current_best_loss = 100
                model_test.set_random_idp(sess=sess,probs=spec)
                tidp = model_test.loss_counter
                print("=====< Start to optimize IDP = %d >=====" % (trained_idp[tidp]*100))
            
            if all_epoch < 0:
                go_on = False
                break   
        r2 = pd.DataFrame.from_dict(r2_dict)
        r2.to_csv(config['r2_file'],index=None)
        log.to_csv(config['log_file'],index=None)
        res = pd.DataFrame.from_dict({'profile':profile_arr,'IDP':idp_arr,'accu':accu_arr})
        res.to_csv(config['result_file'],index=None)
        print(res)
