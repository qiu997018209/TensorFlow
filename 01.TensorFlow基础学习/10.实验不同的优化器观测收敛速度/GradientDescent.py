#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
minist = input_data.read_data_sets(r"C:\Users\vcyber\Desktop\MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

#每次处理100张图片
bach_num = 100
each_batch = minist.train.num_examples // bach_num

#构建神经网络
W_Input = tf.Variable(tf.zeros([784,10]))
B_Input = tf.Variable(tf.zeros([10]))
Output = tf.matmul(x,W_Input)+B_Input
predict = tf.nn.softmax(Output)

#定义损失函数:交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = predict))
#梯度下降优化器
Optimizer = tf.train.GradientDescentOptimizer(0.2)
train_step = Optimizer.minimize(loss)

init = tf.global_variables_initializer()
#将结果放在一个bool型列表中,1代表以列来取最大值下标
result = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))#argmax返回一维张量中最大的值所在的位置
#求准确率:先将bool型转化为float型数据
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #迭代200次
    for epoch in range(20):
        for batch in range(each_batch):
            #每次处理100张图片
            barch_x,barch_y = minist.train.next_batch(each_batch)
            sess.run(train_step,feed_dict={x:barch_x,y:barch_y})

        acc = sess.run(accuracy,feed_dict={x:minist.test.images,y:minist.test.labels})
        print ("Iter:"+str(epoch)+",accuracy:"+str(acc))





