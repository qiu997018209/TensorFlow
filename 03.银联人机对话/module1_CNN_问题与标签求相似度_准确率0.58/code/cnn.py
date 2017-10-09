#coding:utf-8
import tensorflow as tf
import numpy as np

##########################################################################
#  embedding_lookup + cnn + cosine margine ,  batch
##########################################################################
class InsQACNN(object):
    def __init__(
      self, sequence_length, batch_size,
      vocab_size, embedding_size,
      filter_sizes, num_filters, threshold,l2_reg_lambda=0.0):

        #标签
        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        #正确问题
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_2")
        #错误问题
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # Embedding layer
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        #在W中找到input_x_1对应下标的向量组成矩阵返回
        chars_1 = tf.nn.embedding_lookup(W, self.input_x_1)
        chars_2 = tf.nn.embedding_lookup(W, self.input_x_2)
        chars_3 = tf.nn.embedding_lookup(W, self.input_x_3)
        self.embedded_chars_1 = tf.nn.dropout(chars_1, self.dropout_keep_prob)
        self.embedded_chars_2 = tf.nn.dropout(chars_2, self.dropout_keep_prob)
        self.embedded_chars_3 = tf.nn.dropout(chars_3, self.dropout_keep_prob)
        #self.embedded_chars_1 = chars_1
        #self.embedded_chars_2 = chars_2
        #self.embedded_chars_3 = chars_3
        #增加一个维度，-1代表最后一个维，此时的维度为[100,100,100,1],依次是batch_size,句子vector,词vector,新增的维度
        self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)
        self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1)
        self.embedded_chars_expanded_3 = tf.expand_dims(self.embedded_chars_3, -1)

        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        #分布对正向问题，负向问题，标签做卷积和池化
        #filter_sizes为1,2,3,5
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                '''
                4个参数分别为filter_size高h，embedding_size宽w，channel为1，num_filters代表卷积核个数500
                                                列为embedding_size可以保证这个和字向量的大小是一样的，所以对每个句子而言，每个filter出来的结果是一个列向量（而不是矩阵），
                                                列向量再取max-pooling就变成了一个数字，每个filter输出一个数字,num_filters_total个filter出来的结果当然就是[num_filters_total]大小的向量
                                                这样就得到了一个句子的语义表示向量
                '''
                #当filter_size=2时，得到的是[99,1],再做池化的时候得到的是[1,1]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_1,
                    W,#相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                #池化窗口的大小,因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],#[1,x,y,1]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_3,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-3"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-3"
                )
                pooled_outputs_3.append(pooled)
        #500*4
        num_filters_total = num_filters * len(filter_sizes)
        # 扁平化数据,tf.concat在指定维度上连接2个矩阵,pooled_outputs_1里的4个成员为:shape=(?, 1, 1, 500),因此得到的结果就是:shape=(?, 1, 1, 2000)
        # 再做tf.reshape就变成了[batch_size,num_filters_total],每一行代表每一句话的语义表示
        pooled_reshape_1 = tf.reshape(tf.concat(pooled_outputs_1,3), [-1, num_filters_total]) 
        pooled_reshape_2 = tf.reshape(tf.concat(pooled_outputs_2,3), [-1, num_filters_total]) 
        pooled_reshape_3 = tf.reshape(tf.concat(pooled_outputs_3,3), [-1, num_filters_total]) 
        #dropout
        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        pooled_flat_3 = tf.nn.dropout(pooled_reshape_3, self.dropout_keep_prob)
        #计算向量长度
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_3, pooled_flat_3), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) #计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_3), 1)
        #最后的一层并不是通常的分类或者回归的方法，而是采用了计算两个向量（Q&A）夹角的方法
        with tf.name_scope("output"):
            #求两个向量间的夹角
            self.cos_12 = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") #计算向量夹角Batch模式
            self.cos_13 = tf.div(pooled_mul_13, tf.multiply(pooled_len_1, pooled_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(threshold, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            #损失函数的意义就是：让正向答案和问题之间的向量cosine值要大于负向答案和问题的向量cosine值,cosine值越大，两个向量越相近,margin是一个阈值
            #loss = m+self.cos_13-self.cos_12,值越大说明负向问题与答案靠的更近。即小于0是满足条件的
            self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_12, self.cos_13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        

            
