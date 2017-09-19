1.获取数据的地址:http://www.robots.ox.ac.uk/~vgg/data

2.windos下的批量执行文件:retrain.bat
	python C:/Users/vcyber/Desktop/tensorflow-master/tensorflow/examples/image_retraining/retrain.py ^ 			注释:^是连接符,相当于换行,retrain.py是tensorflow源码中的文件tensorflow-master\tensorflow\examples\image_retraining\retrain.py
	--bottleneck_dir bottleneck ^																				注释:指定瓶颈输出目录,是原始的谷歌模型除去最后一层softmax的数据，重新训练其实就是从这里开始
	--how_many_training_steps 200 ^
	--model_dir C:/Users/vcyber/Desktop/Tensorflow/inception_model ^											注释:下载谷歌模型地址,http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
	--output_graph output_graph.pb ^																			注释:最终的训练模型
	--output_labels output_labels.txt ^																			注释:你指定的标签
	--image_dir data/train/																						注释:你指定的数据
	pause