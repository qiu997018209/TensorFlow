#coding:utf-8
'''
Created on 2017年10月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import requests
import json
s = requests

while True:
    method=input('请输入你想测试的方法:')
    if method=='generation' or method=='g':
        quest=input('请输入你想测试的问题:')
        data={"method":'generation','id':1,'jsonrpc':2.0,'params':{'user_id':2,"question":quest}}
    elif method=='train' or method=='t':
        data={"method":'train','id':1,'jsonrpc':2.0}
    elif method=='live' or method=='li':
        data={"method":'live','id':1,'jsonrpc':2.0}
        
    try:
        r = s.post('http://127.0.0.1:8001/deep_generation/v2', json.dumps(data))
        results=json.loads(r.text)
        for quest in results:
            print (quest['results']["questions"])
    except Exception as e:
        print(e)
