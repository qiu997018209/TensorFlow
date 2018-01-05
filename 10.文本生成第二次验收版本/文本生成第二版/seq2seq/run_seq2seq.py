#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import sys
sys.path.append('..')

from data import data
from conf import *
from seq2seq_module import seq2seq
from datetime import timedelta
import tensorflow as tf
import os
import time
import numpy as np

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# ## Train
def start_train(): 
    #配置参数
    best_loss=10 
    require_improvement=2000
    total_batch=-1
    bFlag=False
    start_time = time.time()
      
    if not os.path.exists(args.module_path):
        os.makedirs(args.module_path) 
    
    with tf.Graph().as_default() as g:
        moduls=seq2seq(args,data)          
        with tf.Session(graph=g).as_default() as sess:
            sess.run(tf.global_variables_initializer()) 
            # 保存模型  
            # 配置 Saver
            saver = tf.train.Saver()
            for epoch_i in range(1, args.epochs+1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths,valid_targets_batch,valid_sources_batch,valid_targets_lengths,valid_sources_lengths) in enumerate(
                        data.get_batches()):
                    
                    _, loss = sess.run(
                        [moduls.train_op, moduls.cost],
                        {moduls.inputs: sources_batch,
                         moduls.targets: targets_batch,
                         moduls.learning_rate: args.learning_rate,
                         moduls.target_sequence_length: targets_lengths,
                         moduls.source_sequence_length: sources_lengths})
                     
                    total_batch+=1       
                    if batch_i % args.display_step == 0:
                        '''
                        # 计算validation loss
                        validation_loss = sess.run(
                        [moduls.cost],
                        {moduls.inputs: valid_sources_batch,
                         moduls.targets: valid_targets_batch,
                         moduls.learning_rate: args.learning_rate,
                         moduls.target_sequence_length: valid_targets_lengths,
                         moduls.source_sequence_length: valid_sources_lengths})
                        '''
                        improved_str=''
                        if(loss<best_loss):
                            best_loss=loss
                            saver.save(sess, args.module_path)
                            improved_str='*'
                            last_improved=total_batch
                            
                        time_dif = get_time_dif(start_time)
                        print('Epoch {:>3}/{} Batch {} - Training Loss: {:>6.6f}  - Validation loss: {:>6.6f} Time:{} {}'
                              .format(epoch_i,
                                      args.epochs, 
                                      total_batch, 
                                      loss, 
                                      loss,
                                      time_dif,
                                      improved_str))
    
                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        bFlag=True
                        break  # 跳出循环
                if bFlag==True:
                    print('最低loss值为{}'.format(best_loss))
                    break

def start_test():
    # 输入一句话
    args.mode='test'
    while True:
        input_word = input('请输入您的问题:')
        text = data.source_to_seq(input_word)               
        with tf.Graph().as_default() as g:
            moduls=seq2seq(args,data)            
            with tf.Session(graph=g).as_default() as sess:
                # 加载模型
                saver = tf.train.Saver() 
                saver.restore(sess=sess, save_path=args.module_path)  # 读取保存的模型                        
                predicts = sess.run(moduls.predicts,
                {moduls.inputs: [text]*args.batch_size,
                 moduls.target_sequence_length: [len(input_word)*2]*args.batch_size,
                 moduls.source_sequence_length: [len(input_word)]*args.batch_size})
        
        print(np.shape(predicts)) 
              
        print('  Input Words: {}'.format("".join([data.word_int_to_letter[i] for i in text])))              
        for index,answer in enumerate(predicts[0]):
            #print(answer)
            result='' 
            for i in answer:
                if data.word_int_to_letter[i] == '<EOS>':
                    break
                result+=data.word_int_to_letter[i]
            print('%d:%s'%(index+1,result))
            
if __name__=='__main__':
    args=get_args()
    data=data(args)
    start_train()
    #start_test()
    