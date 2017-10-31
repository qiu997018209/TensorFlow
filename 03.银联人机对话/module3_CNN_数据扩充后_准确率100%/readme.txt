1.将原始数据扩充后，准确率提升到99.8%

2.使用办法:
	启动服务端：python3 http_server.py
	启动客户端测试:python3 http_test.py
	
3.深度学习接口.json:是服务端对外提供的http接口，支持对话，重新训练模型，查询训练进度

4.重要参数说明:
	1.train.py是模型训练功能，默认从Mysql数据库user_id=2中拉取数据，
	2.如果要使用本地数据训练，请将train.py中如也支持处理本地文件,需要令data_help的参数user_id为None,数据格式见:data/data2.txt

5.环境:Python3,tensorflow1.3

6.如果需要扩充数据，详细说明见本仓库:05.data_tool

7.效果见:对话效果.PNG,重新训练效果.PNG,效果是从Mysql中的随机数据中运行，仅供参考
