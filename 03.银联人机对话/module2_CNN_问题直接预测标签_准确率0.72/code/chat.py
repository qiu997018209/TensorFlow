import urllib
import sys
import os
import time
import data_help as dh
import numpy as np
import tensorflow as tf
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

if sys.argv[1:]:
    hostIP = ''
    portNum= int(sys.argv[1])
else:
    hostIP = ''
    portNum = 8000 
    
FLAGS = tf.flags.FLAGS
my_data = dh.data_help()
checkpoint_file = tf.train.latest_checkpoint(os.path.join(my_data.file_path,'runs','checkpoints'))

def start_to_chat(quest_raw):
    begin_time = time.clock()
    quest = my_data.Reload_vocab(quest_raw)          
    predict = list(sess.run(predictions, {input_x: quest, dropout_keep_prob: 1.0}))[0]
    ans = my_data.label_ans[my_data.labels[predict]]
    cost_time = time.clock()-begin_time
    print('AI:chat time:%s,quest:%s,answer:%s'%(cost_time,quest_raw,ans))
    try:
        print('Right:chat time:%s,quest:%s,answer:%s'%(cost_time,quest_raw,my_data.quest_ans[quest_raw]))
    except:
        print('Right:chat time:%s,quest:%s,answer:%s'%(cost_time,quest_raw,"新问题"))
    return ans

class myServer(BaseHTTPRequestHandler):    
    chat_num = 0
    @classmethod
    def count(cls):
        cls.chat_num += 1
        return cls.chat_num
     
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
            post_data = urllib.parse.parse_qs(self.rfile.read(length).decode('utf-8'))           
            #quest = post_data['quest']
            if self.path == "/deep_chat":
                quest = post_data['quest']
                ans = start_to_chat(quest[0])
                self.wfile.write(ans.encode(encoding ='utf_8', errors ='strict'))
            else:
                self.send_error(404, message =None)
            print("chat_num:%d,http cost time:%s"%(self.count(),time.clock()-begin_time))
        except IOError:
            self.send_error(404, message =None)

class ThreadingHttpServer(ThreadingMixIn,HTTPServer):
    pass

if __name__=='__main__':      
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]        
            #启动http服务          
            myServer = ThreadingHttpServer((hostIP,portNum), myServer)
            print("Start to listen on:%s:%d"%(myServer.server_name,myServer.server_port))
            myServer.serve_forever()
            myServer.server_close()

