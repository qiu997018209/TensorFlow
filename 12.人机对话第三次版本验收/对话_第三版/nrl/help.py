#coding:utf-8
'''
Created on 2018年2月7日

@author: qiujiahao

@email:997018209@qq.com

从知识图谱中获取数据
'''
from py2neo import Graph
from collections import defaultdict
class help(object):
    def __init__(self):
        self.graph = Graph('127.0.0.1:7474',user='neo4j',password='123456')
        self.data_process()
        
    def data_process(self):
        out=defaultdict(list)
        result=self.graph.data('match (s)-[r]->(e) return s.name,e.name')
        for item in result:
            out[item['s.name']].append(item['e.name'])
        
        with open('../data/adjlist.txt','w',encoding='utf-8') as f:
            for key,values in out.items():
                num=0
                for value in values:
                    if key != value:
                        num+=1
                        f.write('{} {}'.format(key,value))
                if num !=0:
                    f.write('\n')
            


if __name__=='__main__':
    h=help()