#coding:utf-8
import tensorflow as tf
import numpy as np 

'''
python3:y=wx+b,利用tensorflow学习出w和b
'''
x_data = np.random.rand(100).astype(np.float32)#产生100个0到1之间的随机数,tensorflow中的数据经常使用float32类型
y_data = 0.3*x_data+0.1

####开始深度学习框架####
Weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))#第一个参数代表数据是1维的,-1.0到1.0是随机数的范围
bias=tf.Variable(tf.zeros([1]))

y = x_data*Weight+bias

#损失函数
loss = tf.reduce_mean(tf.square(y-y_data))
#优化器,0.5是学习率,梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)
#最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()
####结束深度学习框架####

sess = tf.Session()
sess.run(init)

for step in range(200):
	#开始训练
	sess.run(train)
	if step%20 == 0:
		#打印当前的w和b值
		print (sess.run(Weight),sess.run(bias))