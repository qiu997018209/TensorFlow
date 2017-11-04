import urllib
import sys
import os
import time
import json 
from threading import Thread
import data_help as dh
import numpy as np
import tensorflow as tf
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

class myServer(BaseHTTPRequestHandler):    
    chat_num = 0
    my_data=''
    input_x=''
    dropout_keep_prob=''
    predictions=''
    @classmethod
    def count(cls):
        cls.chat_num += 1
        return cls.chat_num
    @classmethod
    def train(cls,user_id=2):
        #默认加载data下的数据
        vocab = data.Vocab(FLAGS.vocab_path, 1000000)
        batch_size = FLAGS.beam_size
        hps = seq2seq_attention_model.HParams(
            mode=FLAGS.mode,  # train, eval, decode
            min_lr=0.01,  # min learning rate.
            lr=0.15,  # learning rate
            batch_size=batch_size,
            enc_layers=4,
            enc_timesteps=120,#120
            dec_timesteps=120,#30
            min_input_len=0,  # discard articles/summaries < than this
            num_hidden=256,  # for rnn cell
            emb_dim=128,  # If 0, don't use embedding
            max_grad_norm=2,
            num_softmax_samples=0)  # 4096,If 0, no sampled softmax.
        
        batcher = Batcher(vocab, hps, FLAGS.article_key,
            FLAGS.abstract_key, FLAGS.max_article_sentences,
            FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
            truncate_input=FLAGS.truncate_input)
        tf.set_random_seed(FLAGS.random_seed)        
         
        # Only need to restore the 1st step and reuse it since
        # we keep and feed in state for each step's output.
        decode_mdl_hps = hps._replace(dec_timesteps=1)
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
        decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
        #载入模型
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('No model to decode yet at %s'%(FLAGS.log_root))
    
        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        decoder._saver.restore(sess, ckpt_path) 
        #启动http服务
        start_http_server()
                        
    def __init__(self,request,client_address,hander):
        BaseHTTPRequestHandler.__init__(self,request, client_address,hander) 
    def do_GET(self):
        try:
            self.send_response(200, message =None)
            self.send_header('Content-type','text/html')
            self.end_headers()
            res ='暂时只支持POST请求!!!'
            self.wfile.write(res.encode(encoding ='utf_8', errors ='strict'))
        except IOError:
            self.send_error(404, message =None)

    def do_POST(self):
        try:
            begin_time = time.clock()
            self.send_response(200, message =None)
            self.send_header('Content-type','text/html')
            self.end_headers()
            
            length = int(self.headers['Content-Length'])
            post_data =eval(self.rfile.read(length).decode('utf-8'))         
            #quest = post_data['quest']
            if post_data['method'] == "chat":
                self.process_chat(post_data)
            elif(post_data['method'] == "retrain"):
                self.process_retrain(post_data)
            elif(post_data['method'] == "lookup"):
                self.process_lookup(post_data)
            else:
                self.send_error(404, message =None)
            print("chat_num:%d,http cost time:%s"%(self.count(),time.clock()-begin_time))
        except IOError:
            self.send_error(404, message =None)
    def process_lookup(self,post_data):
        #获取当前进度
        with open(os.path.join(self.my_data.file_path,'process_rate.txt'),'r',encoding='utf-8') as f:
            rate=f.readline().strip().split(':')
            data={}
            params={}
            params['success']='true'
            params['user_id']=post_data['params']['user_id']
            params['progress']=("%d%%")%((int(rate[0])/int(rate[1]))*100)
            params['need_time']=("%d")%(((int(rate[1])-int(rate[0]))*int(rate[2]))/60)
            data['id']=post_data['id']
            data['jsonrpc']=post_data['jsonrpc']
            data['result']=params
            self.wfile.write(json.dumps(data).encode(encoding ='utf_8', errors ='strict'))          
    
    def target(self,post_data):
        user_id=post_data['params']['user_id']
        cmd="python train.py -u %d"%(user_id)
        if(0!=os.system(cmd)):
            print("web端请求重新训练失败")
            return
        #重新载入模型
        myServer.train(user_id) 
                
    def process_retrain(self,post_data):
        #重新开始训练
        user_id=post_data['params']['user_id']         
        t=Thread(target=self.target,args=(post_data,))
        t.start()
        data={}
        params={}
        params['user_id']=user_id
        params['success']="true"
        params['message']="retrain start"
        data['id']=post_data['id']
        data['jsonrpc']=post_data['jsonrpc']
        data['result']=params
        self.wfile.write(json.dumps(data).encode(encoding ='utf_8', errors ='strict'))
                          
    def process_chat(self,post_data):
        quest = post_data['params']['quest']    
        quest = myServer.my_data.Reload_vocab(quest)          
        predict = list(myServer.sess.run(myServer.predictions, {myServer.input_x: quest, myServer.dropout_keep_prob: 1.0}))[0]
        answer = myServer.my_data.labels[predict]
        data={}
        params={}
        params['quest']=post_data['params']['quest']
        params['answer']=answer
        params['user_id']=post_data['params']['user_id']
        params['success']="true"
        data['id']=post_data['id']
        data['jsonrpc']=post_data['jsonrpc']
        data['result']=params     
        self.wfile.write(json.dumps(data).encode(encoding ='utf_8', errors ='strict'))


class ThreadingHttpServer(ThreadingMixIn,HTTPServer):
    pass

             
def main():      
    #启动http服务   
    myServer.train()      
    Server = ThreadingHttpServer(('',8000), myServer)
    print("Start to listen on:%s:%d"%(Server.server_name,Server.server_port))
    Server.serve_forever()
    Server.server_close()    

if __name__ == '__main__':
    main()