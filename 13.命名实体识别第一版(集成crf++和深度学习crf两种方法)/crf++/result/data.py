right=0
wrong=0

with open('out.txt','r',encoding='utf-8') as f:
	for l in f.readlines():
		line=l.strip().split('\t')
		if len(line) != 3:
			continue
		if line[1] == line[2]:
			right+=1
		else:
			wrong+=1
print('准确率:%f'%(right/wrong))