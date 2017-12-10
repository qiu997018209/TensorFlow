#coding:utf-8
'''
Created on 2017年10月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import requests
import json
import xlrd 
import data_help 

s = requests
data_lookup={"method":'lookup','id':1,'jsonrpc':2.0,'params':{'user_id':2}}
data_chat={"method":'chat','id':1,'jsonrpc':2.0,'params':{'user_id':2,"quest":"银联二维码支付怎么用"}}
data_retrain={"method":'retrain','id':1,'jsonrpc':2.0,'params':{'user_id':2}}
right=0
total=0

def send_json(quest):
    data_chat={"method":'chat','id':1,'jsonrpc':2.0,'params':{'user_id':2,"quest":quest}}
    r = s.post('http://127.0.0.1:8000/deep_chat', json.dumps(data_chat))
    r.encoding = 'utf-8'
    ans = eval(r.text)   
    return ans['result']['answer']
 
my_data=data_help.data_help(bProcessData=False,user_id=2)      
for key,values in my_data.temp.items():
    for value in values:
        total+=1
        if(values.index(value)==0):
            right_answer=send_json(value)
        else:
            if right_answer == send_json(value):
                right+=1
        
print("%d/%d"%(right,total))   
                
                