#coding:utf-8
#! /usr/bin/env python3.4
import os
import time
import datetime
import cnn
import operator
import config
import data_help 
import shutil
import random
import tensorflow as tf
import numpy as np

my_data = data_help.data_help()
#生成测试和训练数据
my_data.write_test_train_file()

                                    
with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        #log_device_placement:是否打印设备分配日志
        #allow_soft_placement:如果你指定的设备不存在，允许TF自动分配设备
        session_conf = tf.ConfigProto(
          allow_soft_placement=my_data.Flags.allow_soft_placement,
          log_device_placement=my_data.Flags.log_device_placement)
        
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            module = cnn.InsQACNN(
                sequence_length=my_data.Flags.sequence_length,#256
                batch_size=my_data.Flags.batch_size,#100
                vocab_size=len(my_data.words_dic),
                embedding_size=my_data.Flags.embedding_dim,#128
                filter_sizes=list(map(int, my_data.Flags.filter_sizes.split(","))),
                num_filters=my_data.Flags.num_filters,
                threshold =my_data.Flags.threshold,
                l2_reg_lambda=my_data.Flags.l2_reg_lambda)
    
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-1)
            #optimizer = tf.train.GradientDescentOptimizer(1e-2)
            grads_and_vars = optimizer.compute_gradients(module.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)        
            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(my_data.Flags.file_path, "runs"))
            if(os.path.exists(out_dir)):
                shutil.rmtree(out_dir)
            print("Writing to {}\n".format(out_dir))      
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", module.loss)
            acc_summary = tf.summary.scalar("accuracy", module.accuracy)        
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)        
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            
            def train_step():
                """
                A single training step
                """
                x_batch_1, x_batch_2, x_batch_3 = my_data.get_batch_train_data()
                feed_dict = {
                  module.input_x_1: x_batch_1,
                  module.input_x_2: x_batch_2,
                  module.input_x_3: x_batch_3,
                  module.dropout_keep_prob: my_data.Flags.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, module.loss, module.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                
            def dev_step():
                #随机选100个测试集合来对话
                for _ in range(10):
                    rand_num = random.randint(0,len(my_data.test_quests)-1) 
                    quest = my_data.test_quests[rand_num]
                    my_data.record_data['cost_time'] = time.clock()
                    x_test_1, x_test_2, x_test_3 = my_data.get_test_data(quest)
                    feed_dict = {
                        module.input_x_1: x_test_1,
                        module.input_x_2: x_test_2,
                        module.input_x_3: x_test_3,
                        module.dropout_keep_prob: 1.0
                    }        
                    scores = list(sess.run(module.cos_12, feed_dict))
                    my_data.show_test_result(scores,quest)
            for i in range(my_data.Flags.num_epochs):
                try:                    
                    train_step()
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % my_data.Flags.evaluate_every == 0:
                        print("\nEvaluation:")
                        #dev_step()
                        print("")
                    if current_step % my_data.Flags.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                except Exception as e:
                    print(e) 
                