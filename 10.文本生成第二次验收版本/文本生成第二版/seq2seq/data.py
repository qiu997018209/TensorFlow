#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import numpy as np
import time

class data():
    def __init__(self,args):
        self.args=args
        self.data_process()
        
    def data_process(self):
        data=[]
        with open('../data/train.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\t')
                data.append((self.clean_str(line[0]),self.clean_str(line[1])))
        source_data,target_data=zip(*data)
        # 构造映射表
        self.word_int_to_letter, self.word_letter_to_int = self.extract_character_vocab(source_data+target_data)
        np.random.shuffle(data)
        source_data,target_data=zip(*data)                                   
        # 对字母进行转换
        self.source_int = [[self.word_letter_to_int.get(letter, self.word_letter_to_int['<UNK>']) 
                       for letter in line] for line in source_data]
        self.target_int = [[self.word_letter_to_int.get(letter, self.word_letter_to_int['<UNK>']) 
                       for letter in line] + [self.word_letter_to_int['<EOS>']] for line in target_data] 

        print('seq2seq 数据总量:{},词汇表大小:{}'.format(len(source_data),len(self.word_letter_to_int)))
            
    def extract_character_vocab(self,data):
        '''
                        构造映射表
        '''
        set_words=[]
        special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
        for line in data: 
            for character in line:
                if  character not in set_words:
                    set_words.append(character)
        #set_words = list(set([character for line in data.split('\n') for character in line]))
        # 这里要把四个特殊字符添加进词典
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    
        return int_to_vocab, vocab_to_int
    
    
    # ## Batches
    def pad_sentence_batch(self,sentence_batch, pad_int):
        '''
            对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length      
            参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


    def get_batches(self):
        '''
                定义生成器，用来获取batch
        '''  
        # 将数据集分割为train和validation
        train_source = self.source_int[self.args.batch_size:]
        train_target = self.target_int[self.args.batch_size:]
        # 留出一个batch进行验证
        valid_source = self.source_int[:self.args.batch_size]
        valid_target = self.target_int[:self.args.batch_size]

        pad_valid_sources = np.array(self.pad_sentence_batch(self.source_int, self.word_letter_to_int['<PAD>']))
        pad_valid_targets = np.array(self.pad_sentence_batch(valid_target, self.word_letter_to_int['<PAD>']))
 
        #记录每条记录的长度
        valid_targets_lengths = []
        for target in valid_target:
            valid_targets_lengths.append(len(target))
        
        valid_source_lengths = []
        for source in valid_source:
            valid_source_lengths.append(len(source))
  
        for batch_i in range(len(train_source)//self.args.batch_size):
            start_i = batch_i * self.args.batch_size
            end_i = (batch_i + 1) * self.args.batch_size
            sources_batch = train_source[start_i:end_i]
            targets_batch = train_target[start_i:end_i]
            # 补全序列
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, self.word_letter_to_int['<PAD>']))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, self.word_letter_to_int['<PAD>']))
            
            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            
            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))
            
            yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths,pad_valid_targets,pad_valid_sources,valid_targets_lengths,valid_source_lengths
            
    # ## 预测
    def source_to_seq(self,text):
        '''
                    对源数据进行转换
        '''
        sequence_length = 7
        return [self.word_letter_to_int.get(word, self.word_letter_to_int['<UNK>']) for word in self.clean_str(text)] + [self.word_letter_to_int['<PAD>']]*(sequence_length-len(text))

    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string                   