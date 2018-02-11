#coding:utf-8
'''
Created on 2018年2月7日

@author: qiujiahao

@email:997018209@qq.com

'''
import networkx as nx 
import numpy as np
import random
from gensim.models import Word2Vec

random.seed(32)
np.random.seed(32)

class deepwalk(object):
    def __init__(self):
        self.data_process()
        
    def data_process(self):
        #初始化操作
        self.init()
        
        #模拟随机游走
        self.simulate_walks()
        
        #训练word2vec
        self.train_word2vec()
         
    #初始化操作
    def init(self): 
        #读入数据文件,创建有向图的邻接矩阵
        self.Graph = nx.read_adjlist('../data/adjlist.txt', create_using=nx.DiGraph())
        #设置边的权重
        for i, j in self.Graph.edges():
            self.Graph[i][j]['weight'] = 1.0 
    
    #模拟随机游走    
    def simulate_walks(self, num_walks=10, walk_length=80):
        nodes = list(self.Graph.nodes())
        self.walks=[]
        for walk_index in range(num_walks):
            print('开始第{}/{}次游走'.format(walk_index,num_walks))
            np.random.shuffle(nodes)
            for node in nodes:
                self.walks.append(self.deep_walk(walk_length,node))#记录每一条路径
        
        print('最终存在{}条游走路径'.format(len(self.walks)))
    #游走    
    def deep_walk(self,walk_length,start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.Graph.neighbors(cur))#此处的neighbor是有向的，只返回你指向的neighbor
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
    
    #训练word2vec
    def train_word2vec(self):
        kwargs={}
        kwargs["workers"]=8#设置几个工作线程来训练模型
        kwargs["sentences"] = self.walks
        kwargs["min_count"] = 0 #设置最低有效词频
        kwargs["size"] = 128 #是特征向量的维数
        kwargs["sg"] = 1 #定义训练算法,默认是sg=0,采用CBOW，否则sg=1采用skip-gram

        print ("Start Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        '''
        for word in self.Graph.nodes():
            self.vectors[word] = word2vec[word]
        '''
        word2vec.wv.save_word2vec_format('../data/deepwalk.txt')
        
               
if __name__ == '__main__':
    d=deepwalk()