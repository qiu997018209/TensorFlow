# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

from conf import *
from seq2seq.data import data as seq2seq_data
import tensorflow as tf
from app import app


if __name__ == '__main__':
    args=get_args()
    print('参数列表:\n{}'.format(args))
    print('\nhttp_host:{},http_port:{}'.format(args.http_host,args.http_port))
    app.run(debug=True, host=args.http_host, port=args.http_port)
