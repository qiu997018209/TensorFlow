#coding:utf-8
'''
Created on 2017年12月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import tensorflow  as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework.ops import device

class seq2seq():
    def __init__(self,args,data):
        self.data=data
        self.args=args
        self.build_module()
    #构建模型
    def build_module(self):            
        # 获得模型输入    
        self.add_inputs()
        #创建模型
        self.seq2seq_model()    
                
        #这个操作和one hot也很像，但是指定的不是index而是从前到后有多少个True，返回的是True和False。
        masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
  
        with tf.name_scope("optimization"):
            
            # Loss function
            self.cost = tf.contrib.seq2seq.sequence_loss(
                self.training_logits,
                self.targets,
                masks)
            #学习率衰减
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.96, staircase=True)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
    
            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients,global_step=self.global_step)
 
    # ### Seq2Seq
    # 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型 
    def seq2seq_model(self):
        
        # 获取encoder的状态输出
        _, self.encoder_state = self.get_encoder_layer()
        
        # 预处理后的decoder输入
        self.decoder_input = self.process_decoder_input(self.targets, self.data.word_letter_to_int,self.args.batch_size)
        
        # 将状态向量与输入传递给decoder
        self.decoding_layer() 

    # ## Encoder
    # 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
    # 它会对每个batch执行embedding操作。
    def get_encoder_layer(self):
    
        '''
                构造Encoder层    
                参数说明：
        - input_data: 输入tensor
        - rnn_size: rnn隐层结点数量
        - num_layers: 堆叠的rnn cell数量
        - source_sequence_length: 源数据的序列长度
        - source_vocab_size: 源数据的词典大小
        - encoding_embedding_size: embedding的大小
        '''
        # Encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs, len(self.data.word_letter_to_int), self.args.encoding_embedding_size)
    
        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            single_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.drop_out)
            return single_cell
    
        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.args.rnn_size) for _ in range(self.args.num_layers)])
        
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                          sequence_length=self.source_sequence_length, dtype=tf.float32)
    
        return encoder_output, encoder_state    
    
       
    def decoding_layer(self):
        '''
            构造Decoder层
        
            参数：
        - target_letter_to_int: target数据的映射表
        - decoding_embedding_size: embed向量大小
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
        '''
        # 1. Embedding
        decoder_embeddings = tf.Variable(tf.random_uniform([len(self.data.word_letter_to_int), self.args.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_input)
              
        # 2. 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            single_cell=tf.contrib.rnn.DropoutWrapper(decoder_cell,output_keep_prob=self.drop_out)
            return single_cell
        
        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(self.args.rnn_size) for _ in range(self.args.num_layers)])
         
        # 3. Output全连接层
        output_layer = Dense(len(self.data.word_letter_to_int),
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    
        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)
            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                               training_helper,
                                                               self.encoder_state,
                                                               output_layer) 
            #tf.contrib.seq2seq.dynamic_decode执行decode,最终返回:(final_outputs, final_state, final_sequence_lengths)
            self.training_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  maximum_iterations=self.max_target_sequence_length)

            #tf.identity是返回了一个一模一样新的tensor
            self.training_logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
        # 5. Predicting decoder
        # Replicate encoder infos beam_width times
        if (self.args.mode=='test'):
            with tf.variable_scope("predict"):
                decoder_initial_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.args.beam_size)
                start_tokens = tf.tile(tf.constant([self.data.word_letter_to_int['<GO>']], dtype=tf.int32), [self.args.batch_size], 
                                           name='start_tokens')
                # Define a beam-search decoder
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=decoder_embeddings,
                        start_tokens=start_tokens,
                        end_token=self.data.word_letter_to_int['<EOS>'],
                        initial_state=decoder_initial_state,
                        beam_width=self.args.beam_size,
                        output_layer=output_layer,
                        length_penalty_weight=0.0) 
        
                # Dynamic decoding
                self.predict_decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                     maximum_iterations=self.max_target_sequence_length)
                                                                                                
                self.predicts = tf.identity(tf.transpose(self.predict_decoder_outputs.predicted_ids, perm=[0, 2, 1]),'predictions')            
     
    def add_inputs(self):
        '''
                        模型输入tensor
        '''
        self.inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.drop_out = tf.placeholder(tf.float32, name='drop_out')
        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
    # ## Decoder
    # ### 对target数据进行预处理
    def process_decoder_input(self,data, vocab_to_int, batch_size):
        '''
                        补充<GO>，并移除最后一个字符
        '''
        # tf.stride_slice(data, begin, end,stride)
        # cut掉最后一个字符
        # 和tf.slice的区别是slice的end索引是闭区间，stride_slice的end索引是开区间，所以一个截掉最后一列(remove the last element or column of each line)的小技巧是用stride_slice(data, [0, 0], [rows, -1])，
        # 但是如果是用slice(data, [0, 0], [rows, -1])则仍是原矩阵。
    
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        #tf.fill(shape,value,name=None)创建一个形状大小为shape的tensor，初始值为value
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    
        return decoder_input 

    def get_device(self):
        if self.args.gpu:
            return '/gpu:0'
        else:
            return '/cpu:0'