#coding:utf-8
'''
Created on 2017年9月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import os

import data_help_lib 

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
    def __init__(self,file_path='data/yinlian',bProcessData=True,user_id=None):
        data_help_lib.my_lib.__init__(self)
        self.file_path=file_path       
        #开始处理数据
        if(bProcessData==True):
            self.start_process(user_id) 
    def start_process(self,user_id):   
        #处理数据
        self.data_process(user_id)
        #处理同义词词表
        self.process_same_words()
        #处理句式
        self.process_sentences()
        #生成词袋
        self.build_word_dic()
        #生成新的数据标签集合，供其它模型使用
        self.write_new_data()
        #生成测试和训练集
        #self.write_train_test()
        #self.write_train_test2()
        self.write_train_test3()
                           
if __name__=="__main__":
    my_data = data_help()
 

    