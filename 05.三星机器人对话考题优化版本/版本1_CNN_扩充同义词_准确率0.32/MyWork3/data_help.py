#coding:utf-8
'''
Created on 2017年9月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import numpy as np
import time
import random
import os
import tensorflow as tf
import pickle
import data_help_lib
from tensorflow.contrib import learn
'''
本文件处理数据
entry {
    question ID:0
    question :﻿IC卡闪付是什么
    answer ID:0
    answer :银联金融IC卡——IC卡是集成电路卡（IntegratedCircuitCard）的英文简称，也称之为智能卡、芯片卡等。银联金融IC卡符合国际EMV统一标准及安全要求，具有安全、快捷、多应用的特性。“闪付”——“闪付”是银联金融IC卡的一种快捷交易方式。使用“闪付”功能时，只需将卡片靠近POS机的“闪付”感应区（即“挥卡”），即可快速完成交易。
    label ID:0
    label :什么是IC卡闪付？
    subject ID:0
    subject:银联二期_基本业务_IC卡业务_FAQ
}
'''
class data_help(data_help_lib.my_lib):
    def __init__(self,file_path='data/',bProcessData=True,user_id=None):
        data_help_lib.my_lib.__init__(self)
        self.file_path=file_path 
        #开始处理数据
        if(bProcessData==True):
            self.start_process(user_id) 
    #处理数据文件
    def start_process(self,user_id):
        #初始化
        self.Init(user_id)
        #获取与解析数据
        self.data_process(user_id)              
        #回退相关资源
        self.Fini(user_id)
        
if __name__=="__main__":
    my_data = data_help()

    