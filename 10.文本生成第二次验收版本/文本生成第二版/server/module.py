# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
from seq2seq.seq2seq_module import seq2seq 
       
class seq2seq_module():
    def __init__(self,args,seq2seq_data):
        self.args=args
        self.data=seq2seq_data 
        args.mode='test'            
        with tf.Graph().as_default() as g:
            self.module=seq2seq(self.args,self.data)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer()) 
                saver = tf.train.Saver()
                saver.restore(sess=self.session, save_path=args.module_path)  # 读取保存的模型
        print('seq2seq模型加载完毕')
        
    def predict(self,quest):
        print('seq2seq预测问题:{}'.format(quest))
        text = self.data.source_to_seq(quest)        
        predicts = self.session.run(self.module.predicts, {self.module.inputs: [text]*self.args.batch_size, 
                                              self.module.target_sequence_length: [len(quest)*2]*self.args.batch_size, 
                                              self.module.source_sequence_length: [len(quest)]*self.args.batch_size})

        seq2seq_answer=[]
        for index,answer in enumerate(predicts[0]):
            #print(answer)
            result='' 
            for i in answer:
                if self.data.word_int_to_letter[i] == '<EOS>':
                    break
                result+=self.data.word_int_to_letter[i]
            seq2seq_answer.append(result)
            print('%d:%s'%(index+1,result))        
        return seq2seq_answer
    
def quest_genara(params):
    #cnn_module=params[3]
    seq2seq_module=params[2]
    #cnn_answer=cnn_module.predict(params[0])
    seq2seq_answer=seq2seq_module.predict(params[0])
    
    return seq2seq_answer
