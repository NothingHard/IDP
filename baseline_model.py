import time
import numpy as np
import tensorflow as tf

class baseline_model():
    def __init__(self,params=None):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.learning_rate = params['learning_rate']
        self.hidden = params['hidden']
        
        self.W1 = tf.get_variable(initializer=tf.random_uniform([784, self.hidden],-0.1,0.1),name="W1")
        self.b1 = tf.get_variable(initializer=tf.zeros([self.hidden])+0.1,name="b1")
        self.W2 = tf.get_variable(initializer=tf.random_uniform([self.hidden, self.hidden],-0.1,0.1),name="W2")
        self.b2 = tf.get_variable(initializer=tf.zeros([self.hidden])+0.1,name="b2")
        self.W3 = tf.get_variable(initializer=tf.random_uniform([self.hidden, 10],-0.1,0.1),name="W3")
        self.b3 = tf.get_variable(initializer=tf.zeros([10])+0.1,name="b3")
        
        z1 = tf.add(tf.matmul(self.x,self.W1),self.b1)
        y1 = tf.nn.relu(z1)
        z2 = tf.add(tf.matmul(y1,self.W2),self.b2)
        y2 = tf.nn.relu(z2)
        logits = tf.add(tf.matmul(y2,self.W3),self.b3)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.y)
        loss = tf.reduce_mean(cross_entropy)

        self.pred_probs = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(self.pred_probs,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss = loss
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)