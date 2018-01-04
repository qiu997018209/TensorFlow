#coding:utf-8
'''
Created on 2018年1月3日

@author: qiujiahao

@email:997018209@qq.com

'''
from collections import defaultdict

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
        new_sentence=''
        i=0

        while(i<len(sentence)):
            bfind=False
            j=len(sentence)
            #对于当前的每一个字，从后到前找
            while(i<=j):
                word=''.join(sentence[i:j])
                if self.search(word):
                    new_sentence+='{'+self.word_to_label[word]+'}'
                    #找到了i就往前移动
                    i+=len(word)
                    bfind=True
                    break
                else:
                    #没有找到，j往前移动
                    j-=1
            if bfind==False:
                #从头到尾都没找到，则i向前移动一步
                new_sentence+=sentence[i]
                i+=1 
                                           
        return new_sentence   
        
        

class data(object):
    def __init__(self):
        #字典树
        self.Trie = Trie()
        
        self.data_process()
        
    def data_process(self):
        #载入脱敏词汇表
        self.read_data()
        
        #使用举例
        print(self.Trie.cut('I939手机能下载百度视频播放器吗'))
        print(self.Trie.cut(' 扬声器有沙沙的电流声，如何解决？I9300'))
        print(self.Trie.cut('人体共有多少细胞？细胞一共有多少基因对？多少个基因？'))
          
        #处理脱敏数据
        self.write_data()
    
    def read_data(self):
        #载入脱敏词汇表
        self.label_to_word=defaultdict(set)
        self.word_to_label=defaultdict(str)
        with open('行业敏感数据汇总.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line=='' or line.startswith('#'):
                    continue 
                line=line.split('\t')                
                [self.label_to_word[line[0]].add(word) for word in line[1:]]
        for label,words in self.label_to_word.items():
            for w in words:
                self.word_to_label[w]=label
                #插入字典树
                self.Trie.insert(w)
        #词与标签间的映射关系        
        self.Trie.word_to_label=self.word_to_label
        print('总共有{}个敏感词汇,{}个实体标签'.format(len(self.word_to_label), len(self.label_to_word)))
        print('实体标签是:{}'.format(self.label_to_word.keys()))
        print('敏感词汇是:{}'.format(self.word_to_label.keys()))
    
    def write_data(self):
        with open('origin.txt','r',encoding='utf-8') as f:
            with open('origin2.txt','w',encoding='utf-8') as f2:
                for line in f.readlines():
                    line=line.strip()
                    if line=='' or line.startswith('#'):
                        continue 
                    f2.write('手机'+'\t'+self.Trie.cut(line)+'\t\n\n')             
                    
                           
if __name__=='__main__':
    data=data()
