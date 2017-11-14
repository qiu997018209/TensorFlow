#coding:utf-8
'''
Created on 2017年9月21日

@author: qiujiahao

@email:997018209@qq.com

'''
import time
import os
import tensorflow as tf
import data_help as dh
import numpy as np
import sys
import time
import json
import os
import tensorflow as tf
import data
import seq2seq_attention_decode
import seq2seq_attention_model
import data_help 
import beam_search
import urllib
from collections import namedtuple
from six.moves import xrange
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           'data/my_data_train', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           'data/my_vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'abstract',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', 'result/log', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', 'result/train', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', 'result/eval', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', 'result/decode', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'decode', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 100000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 1,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 5,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 100, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 100, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', True,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('decode_quests_per_ckpt', 10,'Number of batches to decode before restoring next checkpoint')

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')
    
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
            post_data = eval(self.rfile.read(length).decode('utf-8'))         
            #quest = post_data['quest']
            if post_data['method'] == "generation":
                quest = post_data['params']['question']
                print('Server Receive quest:%s'%(quest))
                #开始生成相似的quest
                self.start_generate_text(post_data)
            else:
                self.send_error(404, message =None)
            print("chat_num:%d,http cost time:%s"%(self.count(),time.clock()-begin_time))
        except IOError:
            self.send_error(404, message =None)
 
    def start_generate_text(self,post_data):
        quest=post_data['params']['question']
        (article_batch, _, _, article_lens, _, _, origin_articles,
        origin_abstracts) = self.get_batch_quest(quest)
        bs = beam_search.BeamSearch(
            myServer.decoder._model, myServer.decoder._hps.batch_size,
            myServer.decoder._vocab.WordToId(data.SENTENCE_START),
            myServer.decoder._vocab.WordToId(data.SENTENCE_END),
            myServer.decoder._hps.dec_timesteps)
        questions=[]
        article_batch_cp = article_batch.copy()
        article_batch_cp[:] = article_batch[0]
        article_lens_cp = article_lens.copy()
        article_lens_cp[:] = article_lens[0]
        best_beam = bs.BeamSearch(myServer.sess, article_batch_cp, article_lens_cp)
        print("quest:%s"%(origin_articles[0].replace(' ','')))
        for i in range(len(best_beam)):
            result_beam = best_beam[i]
            decode_output = [int(t) for t in result_beam.tokens[1:]]   
            decoded_output = ''.join(data.Ids2Words(decode_output, myServer.decoder._vocab))
            end_p = decoded_output.find(data.SENTENCE_END, 0)
            if end_p != -1:
                decoded_output = decoded_output[:end_p]
            questions.append(decoded_output)
            print("%doutput:%s"%(i,decoded_output))
        
        posdata={}
        params={}
        params["success"]="true"
        params["user_id"]=post_data['params']['user_id']
        params["questions"]=questions
        posdata['id']=post_data['id']
        posdata['jsonrpc']='2.0'
        posdata['result']=params
        self.wfile.write(json.dumps(posdata).encode(encoding ='utf_8', errors ='strict'))
    @classmethod
    def train(cls):
        cls.vocab = data.Vocab(FLAGS.vocab_path, 1000000)
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
        
        cls.batcher = Batcher(cls.vocab, hps, FLAGS.article_key,
            FLAGS.abstract_key, FLAGS.max_article_sentences,
            FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
            truncate_input=FLAGS.truncate_input)
        tf.set_random_seed(FLAGS.random_seed)        
         
        # Only need to restore the 1st step and reuse it since
        # we keep and feed in state for each step's output.
        decode_mdl_hps = hps._replace(dec_timesteps=1)
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            decode_mdl_hps, cls.vocab, num_gpus=FLAGS.num_gpus)
        cls.decoder = seq2seq_attention_decode.BSDecoder(model, cls.batcher, hps, cls.vocab)
        #载入模型
        cls.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('No model to decode yet at %s'%(FLAGS.log_root))
    
        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        cls.decoder._saver.restore(cls.sess, ckpt_path)
        
    def isChinese(self,word):
        if u'\u4e00' <= word <= u'\u9fff':
            return True
        return False
    
    def get_words(self,sentence):
        words  = ''
        result = []
        for word in sentence:
            #连续的字母和数字视为一个词
            if(self.isChinese(word)):
                if(words != ''):
                    result.append(words)
                    words = '' 
                result.append(word)
            elif word in '@{<“"[(（': 
                if(words != ''):
                    result.append(words)
                    words = '' 
                result.append(word) 
                words=''
            elif word in '}，,.。 >]？?？，）)、/、*"”':
                if(words != ''): 
                    result.append(words)
                    words = ''
                result.append(word) 
                words=''                                             
            else:
                words +=word
        #最后一个字符
        if(words != ''):
            result.append(words)     
        return result
    
    def fill_input_quest(self,quest):
        start_id = myServer.batcher._vocab.WordToId(data.SENTENCE_START)
        end_id = myServer.batcher._vocab.WordToId(data.SENTENCE_END)
        pad_id = myServer.batcher._vocab.WordToId(data.PAD_TOKEN)
        quest = ' '.join(self.get_words(quest)) 
        article_sentences = quest.strip()
        abstract_sentences = article_sentences
        enc_inputs = []
        # Use the <s> as the <GO> symbol for decoder inputs.
        dec_inputs = [start_id]
        enc_inputs += data.GetWordIds(article_sentences, myServer.batcher._vocab)
        dec_inputs += data.GetWordIds(abstract_sentences, myServer.batcher._vocab)
        
        #句子太长
        if len(enc_inputs) > myServer.batcher._hps.enc_timesteps:
            enc_inputs = enc_inputs[:myServer.batcher._hps.enc_timesteps]
        if len(dec_inputs) > myServer.batcher._hps.dec_timesteps:
            dec_inputs = dec_inputs[:myServer.batcher._hps.dec_timesteps]
    
        # targets is dec_inputs without <s> at beginning, plus </s> at end
        #<s>之前额外加上了,此处额外加上</s>
        targets = dec_inputs[1:]
        targets.append(end_id)
        enc_input_len = len(enc_inputs)
        dec_output_len = len(targets)
    
        # 如果比指定长度短，在此处填充,dec_inputs是[<s>,...],targets是[...,<\s>]
        while len(enc_inputs) < myServer.batcher._hps.enc_timesteps:
            enc_inputs.append(pad_id)
        while len(dec_inputs) < myServer.batcher._hps.dec_timesteps:
            dec_inputs.append(end_id)
        while len(targets) < myServer.batcher._hps.dec_timesteps:
            targets.append(end_id)
          
        #将nametupe放入队列之中
        element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                             dec_output_len, article_sentences,abstract_sentences)
        return element   
    
    def get_batch_quest(self,quest):
        #batcher._batch_reader.NextBatch()
        enc_batch = np.zeros(
            (myServer.decoder._hps.batch_size, myServer.decoder._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(
            (myServer.decoder._hps.batch_size), dtype=np.int32)
        dec_batch = np.zeros(
            (myServer.decoder._hps.batch_size, myServer.decoder._hps.dec_timesteps), dtype=np.int32)
        dec_output_lens = np.zeros(
            (myServer.decoder._hps.batch_size), dtype=np.int32)
        target_batch = np.zeros(
            (myServer.decoder._hps.batch_size, myServer.decoder._hps.dec_timesteps), dtype=np.int32)
        loss_weights = np.zeros(
            (myServer.decoder._hps.batch_size, myServer.decoder._hps.dec_timesteps), dtype=np.float32)
        origin_articles = ['None'] * myServer.decoder._hps.batch_size
        origin_abstracts = ['None'] * myServer.decoder._hps.batch_size
        
        for i in xrange(myServer.decoder._hps.batch_size):
            (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
             article, abstract) = self.fill_input_quest(quest)
                   
            origin_articles[i] = article
            origin_abstracts[i] = abstract
            enc_input_lens[i] = enc_input_len
            dec_output_lens[i] = dec_output_len
            enc_batch[i, :] = enc_inputs[:]
            dec_batch[i, :] = dec_inputs[:]
            target_batch[i, :] = targets[:]
            for j in xrange(dec_output_len):
                loss_weights[i][j] = 1
        return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
                loss_weights, origin_articles, origin_abstracts)  
         
class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, vocab, hps,
               article_key, abstract_key, max_article_sentences,
               max_abstract_sentences, bucketing=True, truncate_input=False):
    self._vocab = vocab
    self._hps = hps
    self._article_key = article_key
    self._abstract_key = abstract_key
    self._max_article_sentences = max_article_sentences
    self._max_abstract_sentences = max_abstract_sentences
    self._bucketing = bucketing#批量处理相同长度的
    self._truncate_input = truncate_input
   
class ThreadingHttpServer(ThreadingMixIn,HTTPServer):
    pass
    
                                            
if __name__ == '__main__':    
    #启动http服务    
    myServer.train()      
    Server = ThreadingHttpServer(("",8001), myServer)
    print("Start to listen on:%s:%d"%(Server.server_name,Server.server_port))
    Server.serve_forever()
    Server.server_close()