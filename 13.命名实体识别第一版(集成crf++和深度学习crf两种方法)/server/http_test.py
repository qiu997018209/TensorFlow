#coding:utf-8
'''
Created on 2017年10月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import requests
import json
s = requests

data_ner={"method":'ner','id':1,'jsonrpc':2.0,'params':{'user_id':2,"quest":"我代表中共中央"}}
data_train={"method":'retrain','id':1,'jsonrpc':2.0,'params':{'user_id':2}}
data_live={"method":'live','id':1,'jsonrpc':2.0,'params':{'user_id':2}}


while True:
    method=input('请输入你想测试的方法:')
    if method=='ner' or method=='n':
        data=data_ner
    elif method=='train' or method=='t':
        data=data_train
    elif method=='live' or method=='li':
        data=data_live
    try:
        r = s.post('http://192.168.1.245:1234/deep_ner/v1', json.dumps(data))
        r.encoding = 'utf-8'
        print (r.text)
    except Exception as e:
        print(e)
