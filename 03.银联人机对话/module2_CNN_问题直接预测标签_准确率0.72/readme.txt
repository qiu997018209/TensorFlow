1.环境:Python 3.4.3,Tensorfloa:1.3.0
  原始模型:https://github.com/dennybritz/cnn-text-classification-tf

2.使用效果:效果.PNG
  训练方法:训练方法.PNG

3.代码功能介绍:
	code/text_cnn.py:CNN模型
	code/eval.py:评估
	code/data_help.py:数据预处理
	code/train:训练
	code/chat.py:加载已经训练好的模型,对话提供http服务
	code/http_test.py:作为客户端,测试http服务
	
	代码执行顺序:train.py->chat.py->http_test.py
	
4.数据部分介绍:
	data/runs:tensorflow中保存的训练好的模型
	data/data.txt:银联对话数据,格式如下:
				entry {
				question ID:1
				question :今天刷卡消费，没有打印签购单，银行卡已扣账，我该如何处理？
				answer ID:1
				answer :亲，小U这厢有礼了！您可以建议商户在原POS机上选择重打印签购单，看是否会有签购单打印出来。如果打印出来了，并且您核对了交易金额、交易卡号等都正确的话，表明交易是成功的，请您在签购单上签名即可。如果没有打印出来或者打印出来的是其他持卡人的，建议您请商户使用您的银行卡在原POS机上做一笔“余额查询”交易，交易完毕5-10分钟后，您可以再看一下多扣的钱款是否有返还。如果您还有其他问题，可以致电中国银联7*24小时服务热线95516，客服人员将协助您作进一步核实，不便之处请您见谅噢！
				label ID:1
				label :今天刷卡消费，没有打印签购单，银行卡已扣账，我该如何处理？
				subject ID:1
				subject:银联二期_基本业务_交易查询与差错篇_FAQ
			}
	data/模型参数对准确率影响记录.txt:训练参数记录
	data/test_data.pickle:测试数据
	data/vocab:词向量