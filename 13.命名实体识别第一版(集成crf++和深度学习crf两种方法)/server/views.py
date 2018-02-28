#coding:utf-8
'''
Created on 2018年1月9日

@author: qiujiahao

@email:997018209@qq.com

'''
 
from flask import jsonify
from conf import *
from BiLSTM_CRF.data import data as lstm_data
from flask import Flask
from flask import request,render_template
from server.app import app
import tensorflow as tf
import module
import json 

args=get_args()
print('当前配置参数列表:\n{}'.format(args))
lstm_data=lstm_data(args)
lstm_module=module.lstm_module(args,lstm_data)

@app.route('/deep_ner/v1',methods=["POST"])
def ner():
    client_params=request.get_json(force=True)
    server_param={}
    if client_params['method'] == 'ner':
        lstm_module.predict(client_params,server_param)
    elif client_params['method'] == 'retrain':
        lstm_module.train(client_params,server_param)
    elif client_params['method'] == 'live':
        params={'success':'true'}
        server_param['result']=params
           
    server_param['id']=client_params['id']
    server_param['jsonrpc']=client_params['jsonrpc']
    server_param['method']=client_params['method']
    print(server_param)
    return json.dumps(server_param, ensure_ascii=False).encode("utf-8")
