#coding:utf-8
import tensorflow as tf

#Fetch:同时操作多个op
value1 = tf.constant(2.0)
value2 = tf.constant(3.0)

result = tf.subtract(value1,value2)
update = tf.add(result,value1)

with tf.Session() as sess:
    #Fech
    print (sess.run([result,update]))
  
#Feed:创建占位符  
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

update = tf.multiply(input1,input2)
with tf.Session() as sess:
    #以字典的形式填充数据
    print(sess.run(update,feed_dict={input1:2.0,input2:8.0}))