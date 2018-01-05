#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
from collections import defaultdict
import numpy as np  

class data():
    def __init__(self):
        self.process_data()
        
    def process_data(self):
        self.combina_file()
        self.get_train_test()

    def get_train_test(self):
        with open('data/data.txt','r',encoding='utf-8') as f:
            with open('data/train.txt','w',encoding='utf-8') as f2:
                for line in f.readlines():
                    line=line.strip()
                    if line=='':
                        continue
                    quests = line.split('\t')[1:]
                    if(len(quests)<2):
                        continue 
                    for quest in quests:
                        label=np.random.choice(quests)
                        while(quest==label):
                            label=np.random.choice(quests)
                        f2.write('%s\t%s\n'%(quest,label))
    #合并文件                     
    def combina_file(self):
        self.quest_label={}
        with open('data/data2.txt','r',encoding='utf-8') as f:
            with open('data/data3.txt','r',encoding='utf-8') as f2:
                for line in f.readlines():
                    line=line.strip()
                    if line=='':
                        continue
                    t,quests=line.split('\t')[0],line.split('\t')[1:]
                    if quests[0] not in self.quest_label:
                        self.quest_label[quests[0]]=[t]+quests[1:]
                    else:
                        self.quest_label[quests[0]]+=quests[1:]
                for line in f2.readlines():
                    line=line.strip()
                    if line=='':
                        continue
                    t,quests=line.split('\t')[0],line.split('\t')[1:]
                    if quests[0] not in self.quest_label:
                        self.quest_label[quests[0]]=[t]+quests[1:]
                    else:
                        self.quest_label[quests[0]]+=quests[1:]
                
        with open('data/data.txt','w',encoding='utf-8') as f:
            for quest,values in self.quest_label.items():
                if(len(values)<5):
                    continue
                if '\t' in values:
                    continue
                f.write('%s\t%s\t%s\t%s\t%s\t%s\n'%(values[0],quest,values[1],values[2],values[3],values[4]))
               
if __name__=='__main__':
    data=data()       