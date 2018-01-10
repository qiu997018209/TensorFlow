#coding:utf-8
'''
Created on 2018年1月9日

@author: qiujiahao

@email:997018209@qq.com

'''
 
from flask import jsonify
from conf import *
from seq2seq.data import data as seq2seq_data
from flask import Flask
from flask import request,render_template
from server.app import app
import tensorflow as tf
import module
import json 

args=get_args()
seq2seq_data=seq2seq_data(args)
seq2seq_module=module.seq2seq_module(args,seq2seq_data)

@app.route('/deep_generation/v2',methods=["POST"])
def quest_genaration():
    final_ans={}
    client_params=request.get_json(force=True)
    server_params={}

    if client_params['method']=="generation":
        seq2seq_module.predict(client_params,server_params)
    elif client_params['method']=="train":
        seq2seq_module.train(client_params,server_params)
    elif client_params['method'] == 'live':
        params={'success':'true'}
        server_params['result']=params        
    server_params['id']=client_params['id']
    server_params['jsonrpc']=client_params['jsonrpc']
    server_params['method']=client_params['method']
    print(server_params)
    return json.dumps(server_params, ensure_ascii=False).encode("utf-8")

