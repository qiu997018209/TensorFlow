#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#产生一个1到200之间100个均匀分布的数字,作为每一行，np.newaxi是新起一个维度
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
#噪音数据:从正态（高斯）分布绘制随机样本:loc:此概率分布的均值,scale:此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

#定义神经网络的中间层

#1行10列的随机值
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
bias_L1 = tf.Variable(tf.zeros([1,10]))
output1  = tf.matmul(x,Weight_L1)+bias_L1
#激活函数
L1 = tf.nn.tanh(output1)

#定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
bias_L2 = tf.Variable(tf.zeros([1,1]))
output2 = tf.matmul(L1,Weight_L2)+bias_L2
#预测值
prediction = tf.nn.tanh(output2)

#定义损失函数
loss = tf.reduce_mean(tf.square(prediction-y))
#使用梯度下降方法训练
optimizer = tf.train.GradientDescentOptimizer(0.2)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    predict_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    #散点图
    plt.scatter(x_data,y_data) 
    #红色实线,宽度为5
    plt.plot(x_data,predict_value,'r-',lw=5)
    plt.show()











