模型下载地址:
http://www.fit.vutbr.cz/~imikolov/rnnlm/

1.训练模型
./rnnlm -train data/train -valid data/test -rnnlm model -hidden 32 -rand-seed 1 -debug 2 -bptt 3 -class 200

-train：训练集；
-valid：验证集；
-rnnlm：模型名；

2.测试模型覆盖测试集的情况
./rnnlm -rnnlm model -test test.txt -debug 0 > scores.txt

-test:测试集文件；
-rnnlm：rnnlm模型文件；
注：ppl越小，覆盖越好，准确率越高


3.生成数据
./rnnlm -rnnlm model -gen 100 > gen 
-rnnlm：模型名；
-gen:生成多个词；