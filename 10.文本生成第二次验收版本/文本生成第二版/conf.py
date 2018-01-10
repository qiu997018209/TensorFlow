#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import argparse

def get_args():
    parser = argparse.ArgumentParser() 
    #seq2seq参数
    parser.add_argument('-b', '--batch_size', help='seq2seq batch_size',type=int,default='32')
    parser.add_argument('-es', '--encoding_embedding_size', help='seq2seq encoding_embedding_size',type=int,default='128')
    parser.add_argument('-ds', '--decoding_embedding_size', help='seq2seq decoding_embedding_size',type=int,default='128')
    parser.add_argument('-l', '--learning_rate', help='seq2seq learning_rate',type=float,default='0.001')
    parser.add_argument('-n', '--num_layers', help='seq2seq num_layers',type=int,default='2')
    parser.add_argument('-r', '--rnn_size', help='seq2seq rnn_size',type=int,default='128')
    parser.add_argument('-e', '--epochs', help='seq2seq epochs',type=int,default='5000')
    parser.add_argument('-d', '--display_step', help='seq2seq display_step',type=int,default='50')
    parser.add_argument('-m', '--module_path', help='seq2seq runs/seq2seq/trained_model.ckpt',type=str,default='../runs/seq2seq')
    parser.add_argument('-k', '--topk', help='seq2seq top k answers',type=int,default='10')
    parser.add_argument('-bs', '--beam_size', help='decoder beam_size',type=int,default='200')
    parser.add_argument('-t', '--mode', help='current mode',type=str,default='test',choices=['train','test'])
    parser.add_argument('-dp', '--dropout', help='dropout',type=float,default='0.8')
    parser.add_argument('-g', '--gpu', help='gpu mode',type=int,default=0)
    #http服务的配置
    parser.add_argument('-p', '--http_port', help='http http_port',type=int,default='8080')
    parser.add_argument('-ho', '--http_host', help='http http_host',type=str,default='0.0.0.0')
    parser.add_argument('-u', '--user_id', help='mysql user_id',type=int,default='2')
    
    args = parser.parse_args()
    return args 


    