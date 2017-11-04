#coding:utf-8
'''
Created on 2017年10月18日

@author: qiujiahao

@email:997018209@qq.com

'''
import time
N = 1000
for i in range(N):
    print("进度:{0}%".format(round((i + 1) * 100 / N)), end="\r")
    time.sleep(0.01)
        