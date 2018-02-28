1.原理:Bi_LSTM抽取特征+CRF+Viterbi
  本项目实现了crf++和基于深度学习的命名实体识别两种方式

2.深度学习环境:Python3+Tensorflow1.5

3.数据来源:MSRA的简体中文NER语料

4.深度学习方法参考文章:https://www.cnblogs.com/Determined22/p/7238342.html

5.深度学习使用方法:
	1.cd server
	2.sudo python3 run_server.py -p 1234	这一步启动服务
	3.sudo python3 http_test.py	输入t(代表你要开始训练模型)
	4.输入n(训练完成后代表你要测试命名实体识别功能，此时会对服务端发送已经写好的句子"我代表中共中央"进行命名实体识别)
	
6.深度学习效果:见效果.PNG
	其中:B-PER代表人名的开始，I-PER代表人名，B-ORG代表组织名开始，I-ORG代表组织名，B-LOC代表地名，I-LOC代表地名，O代表非实体

7.准确率对比:crf++的准确率在:48%,深度学习:81%

8.crf相关说明:
	1.软件:windos是crf++/CRF++-0.58.zip，linux下crf++/CRF++-0.58.tar.gz
	2.数据:crf++/data
	3.结果:crf++/result,其中out.txt是测试结果输出，data.py是统计准确率的脚本
	4.crf++安装与使用中的问题记录在有道云笔记中:人工智能/crf++安装与使用