#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
minist = input_data.read_data_sets(r"C:\Users\vcyber\Desktop\MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
drop_out = tf.placeholder(tf.float32)

#每次处理100张图片
bach_num = 100
each_batch = minist.train.num_examples // bach_num

#构建神经网络
W1_Input = tf.Variable(tf.truncated_normal([784,1000], stddev = 0.1))
B1_Input = tf.Variable(tf.zeros([1000])+0.1)
Output1 = tf.matmul(x,W1_Input)+B1_Input
predict1 = tf.nn.tanh(Output1)
L1_drop = tf.nn.dropout(predict1, drop_out)

W2_Input = tf.Variable(tf.truncated_normal([1000,1500], stddev = 0.1))
B2_Input = tf.Variable(tf.zeros([1500])+0.1)
Output2 = tf.matmul(L1_drop,W2_Input)+B2_Input
predict2 = tf.nn.tanh(Output2)
L2_drop = tf.nn.dropout(predict2, drop_out)

W3_Input = tf.Variable(tf.truncated_normal([1500,10], stddev = 0.1))
B3_Input = tf.Variable(tf.zeros([10])+0.1)
Output3 = tf.matmul(L2_drop,W3_Input)+B3_Input
predict3 = tf.nn.tanh(Output3)
predict = tf.nn.dropout(predict3, drop_out)


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
            sess.run(train_step,feed_dict={x:barch_x,y:barch_y,drop_out:0.7})

        test_acc  = sess.run(accuracy,feed_dict={x:minist.test.images,y:minist.test.labels,drop_out:0.7})
        train_acc = sess.run(accuracy,feed_dict={x:minist.train.images,y:minist.train.labels,drop_out:0.7})
        
        print ("Iter:"+str(epoch)+",test accuracy:"+str(test_acc)+",train accuracy:"+str(train_acc))





