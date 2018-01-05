# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

from conf import *
from cnn.data import data as cnn_data
from seq2seq.data import data as seq2seq_data
from flask import Flask
from flask import request,render_template


import tensorflow as tf
import module
import json  
from flask import jsonify

app = Flask(__name__)

@app.route('/deep_generation/v2',methods=["POST"])
def quest_genara():
    final_ans={}
    seq2seq_answer=['error']
    try:
        quest=request.get_json(force=True)['params']['question']
        seq2seq_answer=module.quest_genara((quest,args,seq2seq_module))
        final_ans["result"]={"success":"true","user_id": request.get_json(force=True)['params']['user_id'],"questions": seq2seq_answer}
    except Exception as e:
        print(e)
        final_ans["result"]={"success":"false","user_id": request.get_json(force=True)['params']['user_id'],"questions": seq2seq_answer}
    final_ans['id']=request.get_json(force=True)['id']
    final_ans['jsonrpc']=request.get_json(force=True)['jsonrpc']
    final_ans['method']="generation"
    print(final_ans)
    return json.dumps(final_ans, ensure_ascii=False).encode("utf-8")

if __name__ == '__main__':
    args=get_args()
    print('参数列表:\n{}'.format(args))
    #cnn_data=cnn_data(args)
    #cnn_module=module.cnn_module(args,cnn_data)
    seq2seq_data=seq2seq_data(args)
    seq2seq_module=module.seq2seq_module(args,seq2seq_data)
    print('\nhttp_host:{},http_port:{}'.format(args.http_host,args.http_port))
    app.run(debug=True, host=args.http_host, port=args.http_port)
