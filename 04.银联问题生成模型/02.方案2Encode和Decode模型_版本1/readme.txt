1.原始模型地址:https://github.com/tensorflow/models/tree/master/research/textsum
  模型原理说明:http://blog.csdn.net/tensorflowshizhan/article/details/69230070

2.环境:tensorflow1.3 python3

3.效果:见图片，效果.png

4.训练:seq2seq_attention.py
  数据格式:data_tool/data/yinlian/my_data_test

5.data_tool是解析与清理数据的,入口代码为:data_help.py
  数据量在20万左右

6.模型说明:Encode + Decode + Attention机制,才用深度双向RNN，LSTM模型，beam_search解码

7.http接口使用说明:
	python3 http_server.py

8.如果需要直接运行，可加载以下我训练好的模型,放在quest_genaration目录下，然后运行http_server.py即可
	http://pan.baidu.com/s/1gfdAKOb

9.总结:本模型可以达到一个问法里一到2个关键词的替换
  优点:1.语法流畅，可读性好。2.其结果受人工编写的句式影响，其结果可控，稳定。
  缺点:1.由于人的思维定式导致多样性差，2.需要人工介入，通用性差
