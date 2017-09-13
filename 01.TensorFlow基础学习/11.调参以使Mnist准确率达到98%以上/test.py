#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
minist = input_data.read_data_sets(r"C:\Users\vcyber\Desktop\MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
lr = tf.Variable(0.001,dtype=tf.float32)
#每次处理100张图片
bach_num = 100
each_batch = minist.train.num_examples // bach_num

#构建神经网络
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
B1 = tf.Variable(tf.zeros([500])+0.1)
result1 = tf.matmul(x,W1)+B1
output1 = tf.nn.tanh(result1)

W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
B2 = tf.Variable(tf.zeros([300])+0.1)
result2 = tf.matmul(output1,W2)+B2
output2 = tf.nn.tanh(result2)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
B3 = tf.Variable(tf.zeros([10])+0.1)
result3 = tf.matmul(output2,W3)+B3
predict = tf.nn.softmax(result3)

#定义损失函数:交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = predict))
#梯度下降优化器
#Optimizer = tf.train.GradientDescentOptimizer(0.2)
#1e-2代表10的负二次方
Optimizer = tf.train.AdamOptimizer(lr)
train_step = Optimizer.minimize(loss)

init = tf.global_variables_initializer()
#将结果放在一个bool型列表中,1代表以列来取最大值下标
result = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))#argmax返回一维张量中最大的值所在的位置
#求准确率:先将bool型转化为float型数据
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #迭代50次
    for epoch in range(50):
        #每次迭代更新学习率
        sess.run(tf.assign(lr, 0.001*(0.95**epoch)))
        
        for batch in range(each_batch):
            #每次处理100张图片
            barch_x,barch_y = minist.train.next_batch(each_batch)
            sess.run(train_step,feed_dict={x:barch_x,y:barch_y})
            
        learning_rating = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:minist.test.images,y:minist.test.labels})
        print ("Iter:"+str(epoch)+",learning rating:"+str(learning_rating)+",accuracy:"+str(acc))





