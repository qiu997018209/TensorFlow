
#coding:utf-8
'''
Created on 2017年11月30日

@author: qiujiahao

@email:997018209@qq.com

'''
#去重
def fun1():
    quest_quests={}
    key=''
    num=0
    with open('data/sentences2.txt','w',encoding='utf-8') as f2:
        with open('data/sentences.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if(line==''):
                    continue
                if line.startswith('#'):
                    key=line
                    if line not in quest_quests:
                        quest_quests[line]=set()
                    else:
                        num+=1
                        print(num)
                else:
                    quest_quests[key].add(line)
        for key in quest_quests:
            f2.write('%s\n'%(key))
            for quest in quest_quests[key]:
                f2.write('%s\n'%(quest))
#找出打错的问题
def fun2():
    wrong_quests=[]
    all_quests={}
    num=0
    with open('data/accuracy.txt','r',encoding='utf-8') as f1:           
        with open('data/origin_test.txt','r',encoding='utf-8') as f2:
            for line in f1.readlines():
                wrong_quests.append(line.strip().split(':')[1])
            for line in f2.readlines():  
                num+=1 
                all_quests[num]=line.strip().split('\t')[0]
                num+=1
                all_quests[num]=line.strip().split('\t')[0]
            
    with open('data/important2.txt','w',encoding='utf-8') as f1: 
        write=[]
        for wrong in wrong_quests:
            quest = all_quests[int(wrong)]
            if(quest not in write):
                write.append(quest)
                f1.write("#"+quest+'\n\n') 

                        
fun1()
#fun2()              
