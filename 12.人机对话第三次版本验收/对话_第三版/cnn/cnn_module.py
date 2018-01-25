#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,args):
        tf.set_random_seed(66)
        self.args=args
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.args.max_document_lenth], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.args.vocab_size, self.args.embedding_size], -1.0, 1.0),
                name="W")
         
            #self.update =tf.assign(self.W,words_embedding)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(list(map(int, self.args.filter_sizes.split(",")))):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.args.embedding_size, 1, self.args.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.args.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.args.max_document_lenth - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.args.num_filters * len(self.args.filter_sizes.split(","))
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 定义attention layer 
        attention_size = num_filters_total
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([num_filters_total, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            #取出的outputs[t]是每个单词输入后得到的结果，[batch_size,2*rnn_size(双向)]
            #因此u_t可以认为是对这个单词结果的打分,[batch_size,attention_size]
            u_t = tf.tanh(tf.matmul(self.h_pool_flat, attention_w) + attention_b) 
            #最终得到的u_list是[sequence_length,batch_size,attention_size],是每一个单词的权重
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            #将权重变为[batch_size,1]
            z_t = tf.matmul(u_t, u_w)
            #取概率权重
            self.alpha = tf.nn.softmax(z_t)
            #[batch_size,num_filters_total]*[batch_size,1]=[batch_size,num_filters_total],实际就是对每一个乘以一个权重
            self.final_output=self.h_pool_flat*self.alpha

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.final_output, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.args.num_class],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.args.num_class]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.scores=tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
       
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.args.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            
        self.optim = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)
            
