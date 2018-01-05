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
    parser.add_argument('-r', '--rnn_size', help='seq2seq rnn_size',type=int,default='50')
    parser.add_argument('-e', '--epochs', help='seq2seq epochs',type=int,default='5000')
    parser.add_argument('-d', '--display_step', help='seq2seq display_step',type=int,default='50')
    parser.add_argument('-m', '--module_path', help='seq2seq runs/seq2seq/trained_model.ckpt',type=str,default='../runs/seq2seq/trained_model.ckpt')
    parser.add_argument('-k', '--topk', help='seq2seq top k answers',type=int,default='10')
    parser.add_argument('-sa', '--seq2seq_ans_num', help='seq2seq seq2seq_ans_num',type=int,default='5')
    parser.add_argument('-bs', '--beam_size', help='decoder beam_size',type=int,default='5')
    parser.add_argument('-bw', '--beam_width', help='decoder beam_width',type=int,default='100')
    parser.add_argument('-t', '--mode', help='current mode',type=str,default='train',choices=['train','test'])

    
    '''
    #cnn参数
    parser.add_argument('-csb', '--save_per_batch', help='cnn save_per_batch',type=int,default='50')
    parser.add_argument('-cp', '--dropout_keep_prob', help='cnn dropout_keep_prob',type=float,default='0.5')
    parser.add_argument('-cpb', '--print_per_batch', help='cnn print_per_batch',type=int,default='50')
    parser.add_argument('-cm', '--cmodule_path', help='cnn print_per_batch',type=str,default='../runs/cnn/trained_model.ckpt')
    parser.add_argument('-cdl', '--max_document_lenth', help='cnn max_document_lenth',type=int,default='50')
    parser.add_argument('-cn', '--num_class', help='cnn num_class',type=int,default='0')
    parser.add_argument('-cv', '--vocab_size', help='cnn vocab_size',type=int,default='5000')
    parser.add_argument('-cb', '--cbatch_size', help='cnn batch_size',type=int,default='64')
    parser.add_argument('-cne', '--num_epochs', help='cnn num_epochs',type=int,default='100')
    parser.add_argument('-ces', '--embedding_size', help='cnn embedding_size',type=int,default='128')
    parser.add_argument('-cfs', '--filter_sizes', help='cnn filter_sizes',type=str,default='1,2,3,4,5')
    parser.add_argument('-cnf', '--num_filters', help='cnn num_filters',type=int,default='128')
    parser.add_argument('-cl', '--clearning_rate', help='cnn learning_rate',type=float,default='0.001')
    parser.add_argument('-cr', '--l2_reg_lambda', help='cnn l2_reg_lambda',type=int,default='0')
    parser.add_argument('-ca', '--cnn_ans_num', help='cnn cnn_ans_num',type=int,default='5')
    ''' 
    #http服务的配置
    parser.add_argument('-p', '--http_port', help='http http_port',type=int,default='8080')
    parser.add_argument('-ho', '--http_host', help='http http_host',type=str,default='0.0.0.0')
    parser.add_argument('-u', '--user_id', help='mysql user_id',type=int,default='2')
    
    args = parser.parse_args()
    return args 


    