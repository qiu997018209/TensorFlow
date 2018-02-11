#coding:utf-8
'''
Created on 2018年2月7日

@author: qiujiahao

@email:997018209@qq.com

'''
import networkx as nx 
import matplotlib.pyplot as plt

G = nx.read_adjlist('../data/adjlist2.txt', create_using=nx.DiGraph())
nx.draw(G) 
plt.show() 

for i, j in G.edges():
    print(i,j)
    G[i][j]['weight'] = 1.0 
for node in G.nodes():
    print(node)
    print('neighbor',list(G.neighbors(node)))
