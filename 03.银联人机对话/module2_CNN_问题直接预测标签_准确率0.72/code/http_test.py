#coding:utf-8
'''
Created on 2017年10月9日

@author: qiujiahao

@email:997018209@qq.com

'''
import requests

s = requests

data={"quest":"银联卡闪付功能，是不是只要卡片上带有闪付标志，就可以直接辉卡交易"}
r = s.post('http://127.0.0.1:8000/deep_chat', data)

print (r.status_code)
print (r.headers['content-type'])
r.encoding = 'utf-8'
print (r.text)