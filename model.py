import time
import numpy as np
import tensorflow as tf

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
def my_activation(x, act_type):
    if act_type=="lrelu":
        return lrelu(x, 0.01)
    elif act_type=="sigmoid":
        return tf.nn.sigmoid(x)
    else:
        return tf.nn.relu(x)

class model():
    def __init__(self,profile,params=None):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.loss_counter = 0
        self.profile = profile
        self.learning_rate = 0.001
        self.alpha = params['alpha']
        self.act_type = params['act_type']

        if params==None:
            self.W1 = tf.get_variable(initializer=tf.random_uniform([784, 100],-0.1,0.1),name="W1")
            self.b1 = tf.get_variable(initializer=tf.zeros([100])+0.1,name="b1")
            self.W2 = tf.get_variable(initializer=tf.random_uniform([100, 100],-0.1,0.1),name="W2")
            self.b2 = tf.get_variable(initializer=tf.zeros([100])+0.1,name="b2")
            self.W3 = tf.get_variable(initializer=tf.random_uniform([100, 10],-0.1,0.1),name="W3")
            self.b3 = tf.get_variable(initializer=tf.zeros([10])+0.1,name="b3")
            self.idp = np.arange(0.1,1.05,0.1)
        else:
            self.W1 = tf.get_variable(initializer=params['W1'],name="W1")
            self.b1 = tf.get_variable(initializer=params['b1'],name="b1")
            self.W2 = tf.get_variable(initializer=params['W2'],name="W2")
            self.b2 = tf.get_variable(initializer=params['b2'],name="b2")
            self.W3 = tf.get_variable(initializer=params['W3'],name="W3")
            self.b3 = tf.get_variable(initializer=params['b3'],name="b3")
            self.idp = params['idp']
            
        def half_exp(n,k=1,dtype='float32'):
            n_ones = int(n/2)
            n_other = n - int(n/2)
            return np.append(np.ones(n_ones,dtype=dtype),np.exp((1-k)*np.arange(n_other),dtype=dtype))

        if profile == "linear":
            self.r1 = tf.get_variable(initializer=np.linspace(1,0,num=784,endpoint=False,dtype='float32'),name="r1",dtype='float32')
            self.r2 = tf.get_variable(initializer=np.linspace(1,0,num=100,endpoint=False,dtype='float32'),name="r2",dtype='float32')
            self.r3 = tf.get_variable(initializer=np.linspace(1,0,num=100,endpoint=False,dtype='float32') ,name="r3",dtype='float32')
        elif profile == "all-one":
            self.r1 = tf.get_variable(initializer=np.ones(784,dtype='float32'),name="r1",dtype='float32')
            self.r2 = tf.get_variable(initializer=np.ones(100,dtype='float32'),name="r2",dtype='float32')
            self.r3 = tf.get_variable(initializer=np.ones(100,dtype='float32'),name="r3",dtype='float32')
        elif profile == "half-exp":
            self.r1 = tf.get_variable(initializer=half_exp(784,2),name="r1",dtype='float32')
            self.r2 = tf.get_variable(initializer=half_exp(100,2),name="r2",dtype='float32')
            self.r3 = tf.get_variable(initializer=half_exp(100,2),name="r3",dtype='float32')
        else:
            self.r1 = tf.get_variable(initializer=np.array(1.0/(np.arange(784)+1),dtype='float32'),name="r1",dtype='float32')
            self.r2 = tf.get_variable(initializer=np.array(1.0/(np.arange(100)+1),dtype='float32'),name="r2",dtype='float32')
            self.r3 = tf.get_variable(initializer=np.array(1.0/(np.arange(100)+1),dtype='float32') ,name="r3",dtype='float32')
        
        ''' non-increasing weight clip '''
        def clip_in_order(last_output,current_input):
            added = tf.cond(current_input > last_output[0], lambda: last_output[0], lambda: current_input)
            return (added,last_output[1])
        clipped_gamma = tf.scan(fn=clip_in_order,
                                elems = self.r2,
                                initializer = (self.r2[0],self.r2[1]))
        self.gamma_clip_manually = tf.assign(ref=self.r2,value=clipped_gamma[0])
        
        ''' trainable variables '''
        tvars_trainable = tf.trainable_variables()
        print(tvars_trainable)
        self.gamma_vars = [self.r1,self.r2,self.r3]
        
        for rm in self.gamma_vars:
            tvars_trainable.remove(rm)
            print('%s is not trainable.'% rm)
        self.tvars_trainable = tvars_trainable
        
        ''' op_list '''
        self.loss_list = []
        self.train_op_list = []
        self.train_op_gamma_list = []
        
        ''' define components used in optimizing at different IDP '''
        loss_ = 0.0
        z1 = tf.add(tf.matmul(self.x,self.W1),self.b1)
        y1 = my_activation(z1,self.act_type)
        for idp in self.idp:
            with tf.variable_scope(self.profile+str(idp),reuse=None):
                n_ones = int(100 * idp)
                n_zeros = 100 - n_ones
                print("ones=%d, zeros=%d" % (n_ones,n_zeros))
                t2 = tf.get_variable(initializer=np.append(np.ones(n_ones,dtype='float32'),np.zeros(n_zeros,dtype='float32')),name="t2",dtype='float32',trainable=False)
                p2 = tf.multiply(self.r2,t2)
                z2 = tf.add(tf.matmul(tf.multiply(p2,y1),self.W2),self.b2)
            y2 = my_activation(z2,act_type=self.act_type)
            logits = tf.add(tf.matmul(y2,self.W3),self.b3)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.y)
            this_loss = tf.reduce_mean(cross_entropy)
            this_op = tf.train.AdamOptimizer(self.learning_rate).minimize(this_loss,var_list=self.tvars_trainable)
            this_op_gamma = tf.train.AdamOptimizer(self.learning_rate).minimize(this_loss,var_list=self.gamma_vars)
            if loss_ == 0:
                loss_ = this_loss
            else:
                loss_ = self.alpha*loss_ + (1.0-self.alpha)*this_loss
            self.loss_list.append(loss_)
            self.train_op_list.append(this_op)
            self.train_op_gamma_list.append(this_op_gamma)

    def set_trained_idp(self, sess, loss_counter):
        ''' calculate crossentropy at different idp level and sum up as the loss '''
        self.loss_counter = loss_counter
        self.loss = self.loss_list[self.loss_counter]
        self.train_op = self.train_op_list[self.loss_counter]
        self.train_op_gamma = self.train_op_gamma_list[self.loss_counter]

        # calculate r1*(W1*x)+b
        z1 = tf.add(tf.matmul(self.x,self.W1),self.b1)
        y1 = my_activation(z1, act_type=self.act_type)
        with tf.variable_scope(self.profile+str(self.idp[self.loss_counter]),reuse=True):
            t2 = tf.get_variable("t2")
            p2 = tf.multiply(self.r2,t2)
            z2 = tf.add(tf.matmul(tf.multiply(p2,y1),self.W2),self.b2)
        y2 = my_activation(z2, act_type=self.act_type)
        logits = tf.add(tf.matmul(y2,self.W3),self.b3)
        probs = tf.nn.softmax(logits)

        correct_prediction = tf.equal(tf.argmax(probs,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def set_random_idp(self,sess,probs=None):
        if probs is not None:
            loss_counter = np.random.choice(range(len(self.loss_list)),p=probs,size=1)[0]
        else:
            loss_counter = np.random.randint(0,len(self.loss_list))    
        self.loss_counter = loss_counter
        self.loss = self.loss_list[self.loss_counter]
        self.train_op = self.train_op_list[self.loss_counter]
        self.train_op_gamma = self.train_op_gamma_list[self.loss_counter]

        # calculate r1*(W1*x)+b
        z1 = tf.add(tf.matmul(self.x,self.W1),self.b1)
        y1 = my_activation(z1,act_type=self.act_type)
        with tf.variable_scope(self.profile+str(self.idp[self.loss_counter]),reuse=True):
            t2 = tf.get_variable("t2")
            p2 = tf.multiply(self.r2,t2)
            z2 = tf.add(tf.matmul(tf.multiply(p2,y1),self.W2),self.b2)
        y2 = my_activation(z2,act_type=self.act_type)
        logits = tf.add(tf.matmul(y2,self.W3),self.b3)
        probs = tf.nn.softmax(logits)

        correct_prediction = tf.equal(tf.argmax(probs,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def train(self,sess,config,gamma_trainable=False,reuse=False,verbose=True):
        
        self.learning_rate = config['learning_rate']

        """ setting from config """
        epochs = config['epochs']
        batch_per_epoch = config['batch_per_epoch']
        batch_size = config['batch_size']
        save_dir = config['save_dir']
        mnist = config['mnist']

        """ record container """
        val_loss_per_epoch = []
        val_accu_per_epoch = []
        loss_per_epoch = []
        accu_per_epoch = []
        gamma_trainable_per_epoch = []
        
        if not reuse:
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                sess.run(v.initializer)
            print("Initialized.")

        """ training """
        for epoch in range(0,epochs):
            start_time = time.time()
            train_loss = []
            train_accu = []
            for batch in range(0,batch_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                if gamma_trainable:
                    _,loss, accu = sess.run([self.train_op_gamma, self.loss, self.accuracy],
                                                feed_dict = {self.x:batch_x, self.y: batch_y})
                    _ = sess.run([self.gamma_clip_manually])

                else:
                    _,loss, accu = sess.run([self.train_op, self.loss, self.accuracy], 
                                                feed_dict = {self.x:batch_x, self.y: batch_y})
                train_loss.append(loss)
                train_accu.append(accu)

            loss_per_epoch.append(np.mean(train_loss))
            accu_per_epoch.append(np.mean(train_accu))
            gamma_trainable_per_epoch.append(gamma_trainable+0)

            """ validation """
            val_loss , val_accu = sess.run([self.loss,self.accuracy],
                                            feed_dict={self.x:mnist.test.images, self.y:mnist.test.labels})
            val_loss_per_epoch.append(val_loss)
            val_accu_per_epoch.append(val_accu) 

            if verbose:
                print("epoch: %2d \t time: %2f \t loss: %2f \t val_loss: %2f \t accu: %2f \t val_accu: %2f"% 
                  (epoch,time.time() - start_time,np.mean(train_loss), val_loss, np.mean(train_accu), val_accu))

        return({'loss':loss_per_epoch, 'val_loss':val_loss_per_epoch,
                'accu':accu_per_epoch, 'val_accu':val_accu_per_epoch,
                'gamma_trainable':gamma_trainable_per_epoch})
    
    def predict(self,sess,config,idp,scope,reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
            # use variable t to control the on/off of neurons
            n_ones = int(784 * idp)
            n_zeros = 784 - n_ones
            t1 = tf.get_variable(initializer=np.append(np.ones(n_ones,dtype='float32'),np.zeros(n_zeros,dtype='float32')),name="t1",dtype='float32')

            n_ones = int(100 * idp)
            n_zeros = 100 - n_ones
            t2 = tf.get_variable(initializer=np.append(np.ones(n_ones,dtype='float32'),np.zeros(n_zeros,dtype='float32')),name="t2",dtype='float32')

            n_ones = int(100 * idp)
            n_zeros = 100 - n_ones
            t3 = tf.get_variable(initializer=np.append(np.ones(n_ones,dtype='float32'),np.zeros(n_zeros,dtype='float32')),name="t3",dtype='float32')

            p1 = tf.multiply(self.r1,t1)
            p2 = tf.multiply(self.r2,t2)
            p3 = tf.multiply(self.r3,t3)
            
            for v in [t1,t2,t3]:
                sess.run(v.initializer)  
            print("Running IDP mechanism {:.0f}%".format(idp*100),end='\r')
            z1 = tf.add(tf.matmul(self.x,self.W1),self.b1)
            y1 = my_activation(z1,act_type=self.act_type)
            z2 = tf.add(tf.matmul(tf.multiply(p2,y1),self.W2),self.b2)
            y2 = my_activation(z2,act_type=self.act_type)
            logits = tf.add(tf.matmul(y2,self.W3),self.b3)
            self.pred_probs = tf.nn.softmax(logits)

            mnist = config['mnist']
            pred_probs = sess.run([self.pred_probs],
                                  feed_dict={self.x:mnist.test.images, self.y:mnist.test.labels})
            return(pred_probs)