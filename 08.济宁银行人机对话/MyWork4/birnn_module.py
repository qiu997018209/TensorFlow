#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import conf

class BiRNN(object):
	"""
	用于文本分类的双向RNN
	"""
	def __init__(self):
					
		embedding_size=conf.embedding_size
		rnn_size=conf.hidden_dim
		layer_size=conf.num_layers 
		vocab_size=conf.vocab_size
		attn_size=conf.attn_size
		sequence_length=conf.max_document_lenth
		n_classes=conf.num_class
		grad_clip=conf.grad_clip
		learning_rate=conf.learning_rate

		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='input_y')

		# 定义前向RNN Cell
		with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
			lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
			lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=self.keep_prob)

		# 定义反向RNN Cell
		with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
			lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
			lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list), output_keep_prob=self.keep_prob)


		with tf.device('/cpu:0'):
			embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name='embedding')
			inputs = tf.nn.embedding_lookup(embedding, self.input_x)

		# self.input_data shape: (batch_size , sequence_length)
		# inputs shape : (batch_size , sequence_length , rnn_size)
		# 本代码中rnn_size与embedding_size相等
		# bidirection rnn 的inputs shape 要求是(sequence_length, batch_size, rnn_size)
		# 因此这里需要对inputs做一些变换
		# 经过transpose的转换已经将shape变为(sequence_length, batch_size, rnn_size)
		# 只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
		inputs = tf.transpose(inputs, [1,0,2])
		# 转换成(batch_size * sequence_length, rnn_size)
		inputs = tf.reshape(inputs, [-1, rnn_size])
		#转换成list,里面的每个元素是(batch_size, rnn_size)
		#tf.split(input, num_split, dimension)：dimension的意思就是输入张量的哪一个维度，如果是0就表示对第0维度进行切割
		inputs = tf.split(inputs, sequence_length, axis=0)
		#此时每次输入到RNN中的是[batch_size,rnn_size（1个单词的词嵌入向量）],每次输出的是[batch_size,2*rnn_size(双向)]
		with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
			outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs, dtype=tf.float32)

		# 定义attention layer 
		attention_size = attn_size
		with tf.name_scope('attention'), tf.variable_scope('attention'):
			attention_w = tf.Variable(tf.truncated_normal([2*rnn_size, attention_size], stddev=0.1), name='attention_w')
			attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
			u_list = []
			for t in range(sequence_length):
				#取出的outputs[t]是每个单词输入后得到的结果，[batch_size,2*rnn_size(双向)]
				#因此u_t可以认为是对这个单词结果的打分,[batch_size,attention_size]
				u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b) 
				u_list.append(u_t)
			#最终得到的u_list是[sequence_length,batch_size,attention_size],是每一个单词的权重
			u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
			attn_z = []
			for t in range(sequence_length):
				#将权重变为[batch_size,1]
				z_t = tf.matmul(u_list[t], u_w)
				#这样得到的列表attn_z是每一个单词的注意力值,它的维度是[sequence_length,batch_size,1]
				attn_z.append(z_t)
			# transform to batch_size * sequence_length
			# 使用tf.concat后维度并没有变化，仍然是3维
			# [batch_size,sequence_length,1]
			attn_zconcat = tf.concat(attn_z, axis=1)
			self.alpha = tf.nn.softmax(attn_zconcat)
			# transform to sequence_length * batch_size * 1 , same rank as outputs
			#最终alpha_trans是[sequence_length, batch_size, 1]
			alpha_trans = tf.reshape(tf.transpose(self.alpha, [1,0]), [sequence_length, -1, 1])
			#outputs是[sequence_length, batch_size, 2*rnn_size]
			#alpha_trans是[sequence_length, batch_size, 1]
			#由于在第0个维度上取和,final_output是[batch_size,2*rnn_size]
			self.final_output = tf.reduce_sum(outputs * alpha_trans, 0)

		# outputs shape: (sequence_length, batch_size, 2*rnn_size)
		fc_w = tf.Variable(tf.truncated_normal([2*rnn_size, n_classes], stddev=0.1), name='fc_w')
		fc_b = tf.Variable(tf.zeros([n_classes]), name='fc_b')

		#self.final_output = outputs[-1]

		# 用于分类任务, outputs取最终一个时刻的输出
		# self.logits是[batch_size，n_classes]
		self.logits = tf.matmul(self.final_output, fc_w) + fc_b
		self.prob = tf.nn.softmax(self.logits)
		self.scores=tf.nn.softmax(self.logits)
		self.loss = tf.losses.softmax_cross_entropy(self.input_y, self.logits)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)

		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.optim = optimizer.apply_gradients(zip(grads, tvars))
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), tf.argmax(self.prob, axis=1)), tf.float32))

