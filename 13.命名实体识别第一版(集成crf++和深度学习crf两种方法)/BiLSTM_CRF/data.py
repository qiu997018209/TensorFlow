#coding:utf-8
'''
Created on 2018年2月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import random
import numpy as np
import sys
sys.path.append('..')
from collections import defaultdict
from collections import Counter
from gensim.models import Word2Vec
from conf import get_args

random.seed(32)
np.random.seed(32)

class data(object):
    def __init__(self,args):
        self.args=args
        self.data_process()
        
    def data_process(self):
        
        #初始化
        self.init()
        
        #建立词汇表
        self.build_vocab_size()
        
        #训练word2vec
        self.train_word2vec()
        
    def init(self):
        #打印参数
        print('当前配置参数列表:\n{}'.format(self.args))
        
        ## tags, BIO
        self.tag2label = {"O": 0,
                     "B-PER": 1, "I-PER": 2,
                     "B-LOC": 3, "I-LOC": 4,
                     "B-ORG": 5, "I-ORG": 6
                     }
        self.label2tag={value:key for key,value in self.tag2label.items()}
        self.quest_label=defaultdict(int)
        self.labels={value:key for key,value in self.tag2label.items()}
        self.train_data=[]
        self.sentences=[]
        sentence=[]
        tag=[]
        print('开始读入数据')
        with open('../data/data.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split(' ')
                if len(line)<=1:
                    self.sentences.append(sentence)
                    self.quest_label[''.join(sentence)]=tag
                    sentence,tag=[],[]
                    continue
                sentence.append(line[0]) 
                tag.append(self.tag2label[line[1]])
      
        print('标签数量:{}'.format(len(self.labels)))
        print('总数据量:{}'.format(len(self.quest_label)))
    
    def get_batch_data(self):
        """生成批次数据"""
        quests,labels=np.array(list(self.quest_label.keys())),np.array(list(self.quest_label.values()))
        shuffle_indices=np.random.permutation(np.arange(len(self.quest_label)))  
        #生成训练数据 
        rate=int(0.9*len(shuffle_indices))                                              
        train_x,train_y=quests[0:rate],labels[0:rate]
        test_x,test_y=quests[rate:],labels[rate:]
        test_x,test_x_lenth,test_y=self.build_vector(test_x,test_y)        
        print('train_x,train_y,test_x,test_y',train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        num_batches_per_epoch = int((len(train_x)-1)/self.args.batch_size) + 1
        print('num_batches_per_epoch:',num_batches_per_epoch)
        for epoch in range(self.args.num_epochs): 
            print('Epoch:', epoch + 1)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.args.batch_size  
                end_index = min((batch_num + 1) * self.args.batch_size, len(train_x))
                batch_x=train_x[start_index:end_index]
                batch_y=train_y[start_index:end_index]
                batch_x,batch_x_lenth,batch_y=self.build_vector(batch_x,batch_y)
                yield batch_x,batch_y,batch_x_lenth,test_x,test_y,test_x_lenth    
        #建立词汇表
    def build_vocab_size(self):
        """根据训练集构建词汇表，存储"""
        all=''.join(self.quest_label.keys())
        counter = Counter(all)
        count_pairs = counter.most_common(5000 - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<UNK>'] + list(words)
        self.word_to_id=dict(zip(words,range(len(words))))
        print('词汇表数量:%d'%(len(words))) 
        self.vocab_size=len(words) 
        
        #向量化
    def build_vector(self,batch_x,batch_y):
        results=[]
        for quest in batch_x:
            vec=[self.word_to_id[w] for w in quest]
            results.append(vec)
        results=self.pad_sequences(results)
        
        labels=[]
        for label in batch_y:
            ll=label+(len(results[0][0])-len(label))*[0]#对应的tag的长度应该是和quest一样，quest填充的部分，一律为0
            labels.append(ll)
        return np.array(results[0]),np.array(results[1]),np.array(labels)
    
        #训练word2vec
    def train_word2vec(self):
        kwargs={}
        
        kwargs["workers"]=8#设置几个工作线程来训练模型
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = 0 #设置最低有效词频
        kwargs["size"] = self.args.embedded_size #是特征向量的维数
        kwargs["sg"] = 0 #定义训练算法,默认是sg=0,采用CBOW，否则sg=1采用skip-gram

        print ("Start training word2vec...")
        word2vec = Word2Vec(**kwargs)
        self.embeddings = [0]*len(self.word_to_id)
        
        for word,value in self.word_to_id.items():
            try:
                self.embeddings[value] = word2vec[word]
            except:
                self.embeddings[value] = np.array([0]*self.args.embedded_size)#<UNK>这个单词是不存在的，设为0
        self.embeddings=np.float32(self.embeddings)#转换格式    
        #将本批次的填充为一个长度
    def pad_sequences(self,sequences, pad_mark=0):
        max_len = max(map(lambda x : len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list
     
if __name__=='__main__':
    args=get_args()
    d=data(args)            