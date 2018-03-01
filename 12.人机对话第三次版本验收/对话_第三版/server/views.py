#coding:utf-8
'''
Created on 2018年1月9日

@author: qiujiahao

@email:997018209@qq.com

'''
 
from flask import jsonify
from conf import *
from log import log
from cnn.data import data as cnn_data
from flask import Flask
from flask import request,render_template
from server.app import app
from threading import Thread
import tensorflow as tf
import module
import json 


args=get_args()
log('当前配置参数列表:\n{}'.format(args))
cnn_data=cnn_data(args)
cnn_module=module.cnn_module(args,cnn_data)


@app.route('/deep_chat/v2',methods=["POST"])
def chat():
    client_params=request.get_json(force=True)
    server_param={}
    if client_params['method'] == 'chat':
        cnn_module.predict(client_params,server_param)
    elif client_params['method'] == 'retrain':
        cnn_module.train(client_params,server_param)
    elif client_params['method'] == 'lookup':
        cnn_module.lookup(client_params,server_param)
    elif client_params['method'] == 'live':
        params={'success':'true'}
        server_param['result']=params
        
    server_param['id']=client_params['id']
    server_param['jsonrpc']=client_params['jsonrpc']
    server_param['method']=client_params['method']
    log(server_param)
    return json.dumps(server_param, ensure_ascii=False).encode("utf-8")
