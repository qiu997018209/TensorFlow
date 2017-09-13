#coding:utf-8
import tensorflow as tf

one = tf.constant(1)
value = tf.Variable(0,name='count')
new_value = tf.add(one,value)
#赋值操作
update = tf.assign(value,new_value)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
	#取值
	print (sess.run(value))
