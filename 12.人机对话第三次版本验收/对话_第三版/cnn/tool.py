#coding:utf-8
'''
Created on 2018年1月3日

@author: qiujiahao

@email:997018209@qq.com

'''
from collections import defaultdict
from log import *

class TrieNode(object):
    def __init__(self):
        self.data = {}
        self.is_word = False

class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    #插入一个单词
    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True
        
    #查找一个词汇是否是一个指定的单词
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中
    
    #动态分词,默认是以长的优先
    def cut(self,sentence):
        sentence=list(sentence)
        word=''
        new_sentence=[]
        i=0

        while(i<len(sentence)):
            bfind=False
            j=len(sentence)
            #对于当前的每一个字，从后到前找
            while(i<=j):
                word=''.join(sentence[i:j])
                if self.search(word):
                    new_sentence.append(word)
                    #找到了i就往前移动
                    i+=len(word)
                    bfind=True
                    break
                else:
                    #没有找到，j往前移动
                    j-=1
            if bfind==False:
                #从头到尾都没找到，则i向前移动一步
                new_sentence.append(sentence[i])
                i+=1 
                                           
        return new_sentence   
        
        

class word_parser(object):
    def __init__(self,data):
        self.data=data
        #字典树
        self.Trie = Trie()
        
        self.data_process()
        
    def data_process(self):
        log('开始为动态分词引擎插入数据')
        #载入脱敏词汇表
        self.insert_data()
         
    
    def insert_data(self):
        for name in self.data.get_industry_name():
            with open(name,'r',encoding='utf-8') as f: 
                for line in f.readlines():
                    line=line.strip().split('=')
                    if len(line)<=1:
                        continue
                    words=line[1].strip().split(";")                    
                    [self.Trie.insert(w) for w in words  if len(words)>1]
    
    def cut(self,sentence):
        return self.Trie.cut(sentence)
                            