#coding:utf-8
'''
Created on 2018年3月1日

@author: qiujiahao

@email:997018209@qq.com

'''
import logging
import conf
import time
import os
from threading import Thread,Timer

dir = conf.get_args().module_path

start_log = False
if not os.path.exists(dir+'/log'):
    os.makedirs(dir+'/log')

def log(info,level='debug'):
    "获取当前时间"
    print(info,start_log)#此处再打印一下，方便我调试
    if start_log==False:
        return
    date=time.strftime("%Y%m%d")
    filename=dir+'/log/'+date+'.txt'
    loc_time=time.strftime("%H:%M:%S")
    if level=='debug':
        logging.basicConfig(filename=filename,level=logging.DEBUG)
        logging.debug('{}:{}'.format(loc_time, info))
    else:
        logging.basicConfig(filename=filename,level=logging.ERROR)
        logging.error('{}:{}'.format(loc_time, info))
    
    #启动一个守护线程，
def log_deamon():
    loc_time=time.strftime("%H")
    if loc_time == '24':
        log('守护线程信息,请忽略')
    t=Timer(3600,log_deamon)
    t.start()#一个小时后再启动    
    
def set_flag(flag):
    global start_log
    start_log=flag    