#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import argparse

def get_args():
    parser = argparse.ArgumentParser() 
    #http服务的配置
    parser.add_argument('-p', '--http_port', help='server端口号',type=int,default='8080')
    parser.add_argument('-ho', '--http_host', help='server地址',type=str,default='0.0.0.0')
    parser.add_argument('-u', '--user_id', help='数据库mysql id',type=int,default='2')
    #cnn参数
    parser.add_argument('-s', '--same', help='是否开启同义词扩展',type=int,default=1)
    parser.add_argument('-lo', '--log', help='开启日志信息',type=int,default=0)
    parser.add_argument('-i', '--industry', help='0代表通用,1代表汽车,2代表银行',type=int,default=0)
    parser.add_argument('-m', '--module_path', help='模型存储地址,格式举例:../data/runs/xxx',type=str,default='../data/runs/cnn')
    parser.add_argument('-dp', '--dropout_keep_prob', help='cnn dropout_keep_prob',type=float,default='0.5')
    parser.add_argument('-pb', '--print_per_batch', help='cnn print_per_batch',type=int,default='50')
    parser.add_argument('-dl','--max_document_lenth', help='cnn max_document_lenth',type=int,default='50')
    parser.add_argument('-n', '--num_class', help='cnn num_class',type=int,default='0')
    parser.add_argument('-v', '--vocab_size', help='cnn vocab_size',type=int,default='5000')
    parser.add_argument('-b', '--batch_size', help='cnn batch_size',type=int,default='32')
    parser.add_argument('-e', '--num_epochs', help='cnn num_epochs',type=int,default='10')
    parser.add_argument('-em', '--embedding_size', help='cnn embedding_size',type=int,default='128')
    parser.add_argument('-fs', '--filter_sizes', help='cnn filter_sizes',type=str,default='1,2,3,4,5')
    parser.add_argument('-nf', '--num_filters', help='cnn num_filters',type=int,default='128')
    parser.add_argument('-l', '--learning_rate', help='cnn learning_rate',type=float,default='0.001')
    parser.add_argument('-l2', '--l2_reg_lambda', help='cnn l2_reg_lambda',type=int,default='0')
    parser.add_argument('-t', '--time', help='leaf time',type=float,default='0')
    parser.add_argument('-r', '--rate', help='train process',type=float,default='1')
    parser.add_argument('--version', help='version',type=str,default='3.0.0')
    args = parser.parse_args()
    return args 


    