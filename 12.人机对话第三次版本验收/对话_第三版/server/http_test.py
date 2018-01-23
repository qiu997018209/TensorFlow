#coding:utf-8
'''
Created on 2017年10月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import requests
import json
s = requests
data_lookup={"method":'lookup','id':1,'jsonrpc':2.0,'params':{'user_id':2}}
data_chat={"method":'chat','id':1,'jsonrpc':2.0,'params':{'user_id':2,"quest":"我卡里一笔消费，如何查询商户名称？",'rate':0.2}}
data_train={"method":'retrain','id':1,'jsonrpc':2.0,'params':{'user_id':2}}
data_live={"method":'live','id':1,'jsonrpc':2.0,'params':{'user_id':2}}


while True:
    method=input('请输入你想测试的方法:')
    if method=='lookup' or method=='l':
        data=data_lookup
    elif method=='chat' or method=='c':
        data=data_chat
    elif method=='train' or method=='t':
        data=data_train
    elif method=='live' or method=='li':
        data=data_live
    try:
        r = s.post('http://127.0.0.1:1111/deep_chat/v2', json.dumps(data))
        print (r.status_code)
        print (r.headers['content-type'])
        r.encoding = 'utf-8'
        print (eval(r.text))
    except Exception as e:
        print(e)
