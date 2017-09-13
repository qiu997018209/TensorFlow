#coding:utf-8

import tensorflow as tf
#一行2列的常量
matrix1 = tf.constant([[3,3]])
#2行1列的常量
matrix2 = tf.constant([[2],[1]])

result = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
	result2 = sess.run(result)
	print(result2)
