import tensorflow as tf
import numpy as np
import random
import copy
from data_reader import data_reader 
from base import BaseClass
from ops import Encoder
import os
from progressbar import ETA, Bar, Percentage, ProgressBar


class Attention(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        self.num_layer = 2 # how many layers for the encoder part.
        self.use_residual_conn = False # if we use resudual connection. 
        self.input_x = tf.placeholder(tf.int32, [None, self.conf.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.conf.num_classes], name="input_y")
        self.label = tf.placeholder(tf.int32, [None], name="label")
  #      self.sequence_length = tf.placeholder(tf.int32, name= "seq_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.instantiate_weights()
   #     self.initializer=tf.random_normal_initializer(stddev=0.1)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        if not os.path.exists(self.conf.sampledir):
            os.makedirs(self.conf.sampledir)
        self.configure_networks()
    
    def configure_networks(self):
        self.build_network()
        variables = tf.trainable_variables()
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', update_ops=[])
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()

    def build_network(self):
        self.input_x_embeded = tf.nn.embedding_lookup(self.Embedding,self.input_x) 
        self.input_x_embeded = tf.multiply(self.input_x_embeded,tf.sqrt(tf.cast(self.conf.d_model,dtype=tf.float32)))
        input_mask = tf.get_variable("input_mask",[self.conf.sequence_length,1],initializer=self.initializer)
        self.input_x_embeded = tf.add(self.input_x_embeded, input_mask)       
        print('gate shape ',self.input_x_embeded.shape) 
        # encoder_class=Encoder(self.conf.d_model,self.conf.d_k,self.conf.d_v,self.conf.sequence_length,self.conf.h,self.conf.batch_size,
        #     self.num_layer,self.input_x_embeded,self.input_x_embeded,dropout_keep_prob=self.dropout_keep_prob,
        #     use_residual_conn=self.use_residual_conn)
        # Q_encoded,K_encoded = encoder_class.encoder_fn()
        # print('After self-attention, the shape is =========================',Q_encoded.get_shape()) # the shape should be batch*length*d-model
        # #then do a maxpooling make it batch*length
        # Q_encoded = tf.expand_dims(Q_encoded, 1)  #[batch, 1, 300, length]
        # Q_pooled = tf.nn.max_pool(Q_encoded, ksize=[1,1,self.conf.sequence_length,1], strides=[1,1,1,1], padding= 'VALID', name="max_pool") # [batch,1,1,d-dimen]
        # print('After pooling, the shape is =========================', Q_pooled.get_shape())
        # Q_squeezed = tf.squeeze(Q_pooled) #[batch, length]
        # print('After squeezing, the shape is =================', Q_squeezed.get_shape())
        # self.logits = tf.matmul(Q_squeezed, self.W_projection)+ self.b_projection
        self.input_x_embeded = tf.contrib.layers.flatten(self.input_x_embeded)
        #self.input_x_embeded = tf.reshape(self.input_x_embeded, [self.conf.batch_size, -1])
        self.logits = tf.contrib.layers.fully_connected(self.input_x_embeded, self.conf.num_classes, activation_fn=None)
        #self.logits = tf.matmul(self.input_x_embeded, self.W_projection)+ self.b_projection
        print('the shape of logits ============', self.logits.get_shape())
   #     self.label = tf.argmax(self.input_y, axis=1) 
        print('the size of label is ============', self.label)
        self.loss = self.get_loss()
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1)
  #      correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32),tf.cast(tf.argmax(self.input_y, axis=1), tf.int32))
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32),tf.cast(self.label, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        
    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/loss', self.loss))
        summarys.append(tf.summary.scalar('/accuracy', self.accuracy))
        summary = tf.summary.merge(summarys)
        return summary

    def get_loss(self, l2_lambda=0.00001):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.label, logits= self.logits)
        loss = tf.reduce_mean(loss)
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
            if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
        return loss+loss_l2


    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.conf.vocab_size, self.conf.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.conf.num_classes, self.conf.embed_size],dtype=tf.float32) #,initializer=self.initializer
            self.W_projection = tf.get_variable("W_projection", shape=[self.conf.embed_size*self.conf.sequence_length, self.conf.num_classes],initializer=self.initializer)
        #     self.W_projection = tf.get_variable("W_projection", shape=[self.conf.d_model, self.conf.num_classes],initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.conf.num_classes])

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint)
        
        data = data_reader()

        # use word2vec pretrained embedding
        vocabulary = data.vocab_processor.vocabulary_
        initW = data.load_word2vec_embedding(vocabulary, self.conf.path_embedding, True)
        #self.Embedding.assign(initW)
     #$   print(initW[0])
        print("finish load the pre_trained embedding==============")
        iterations = 1
        max_epoch = int (self.conf.max_epoch - (self.conf.checkpoint)/ (len(data.y_train)/self.conf.batch_size))
        print("Each epoch we have the number of batches", int (len(data.y_train)/self.conf.batch_size))
        dropout_keep_prob= 0.5
        for epoch in range(self.conf.max_epoch):
            pbar = ProgressBar()            
            for i in pbar(range(int (len(data.y_train)/self.conf.batch_size))):            
                x, y= data.next_batch(self.conf.batch_size)
          #      feed_dict= {self.input_x:x, self.input_y: y, self.dropout_keep_prob:0.5}
                feed_dict= {self.input_x:x, self.label: y, self.dropout_keep_prob:0.5}
                _, loss, summary, accuracy , prediction,label = self.sess.run([self.train_op, self.loss, self.train_summary, self.accuracy, self.prediction, self.label], feed_dict= feed_dict)

                #print("training loss is =============", loss, "  the acc is =============", accuracy)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary, iterations+self.conf.checkpoint)
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations + 1
            if epoch % self.conf.eva_step == 1:
                x_test, y_test = data.next_test_batch(self.conf.batch_size)
            #    feed_dict2= {self.input_x:x_test, self.input_y: y_test, self.dropout_keep_prob:1}
                feed_dict2= {self.input_x:x_test, self.label: y_test, self.dropout_keep_prob:1}
                acc, test_loss = self.sess.run([self.accuracy, self.loss], feed_dict= feed_dict)
                print("For the epoch  ", epoch, " test acc is  ", acc, " loss is ", test_loss)

    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        model_path = checkpoint_path +'-'+str(epoch)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return       
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")

