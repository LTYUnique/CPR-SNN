import os

path = 'processed/ISPY1/pre_post1_pcr/'
with open(path + "train_list.txt", "r") as f: 
    train_list = f.read()
train_list = train_list.split('\n')
with open(path + "test_list.txt", "r") as f: 
    test_list = f.read()
test_list = test_list.split('\n')

data_list = train_list + test_list

#  shuffle
from random import shuffle
for i in range(5):
    shuffle(data_list)

one_fold_len = int(len(data_list)/5)  # 156/5=31
points = [0, one_fold_len, one_fold_len*2, one_fold_len*3, one_fold_len*4, len(data_list)]
str = '\n'
for i in range(5):
    fold_i_test = data_list[points[i]: points[i+1]]
    fold_i_train = data_list[0:points[i]] + data_list[points[i+1]:len(data_list)]
    print(f'fold{i}', len(fold_i_test), len(fold_i_train), len(fold_i_test) + len(fold_i_train))
    f=open(f"{path}shuffle_fold{i}_train.txt","w")
    f.write(str.join(fold_i_train))
    f.close()
    f=open(f"{path}shuffle_fold{i}_test.txt","w")
    f.write(str.join(fold_i_test))
    f.close()






'''
1.
c = [a,b]
np.save("ab.npy", c)
d = np.load('ab.npy')

2.
l=["A","B","C","D"]
str = '\n'
f=open("k3.txt","w")
f.write(str.join(l))
f.close()

3.
with open("k3.txt", "r") as f:  #打开文本  #, encoding='utf-8'
    data = f.read()   #读取文本
data.split('\n')
# ['aa', 'bb', 'dd']
'''