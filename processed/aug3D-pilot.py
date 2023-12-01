import numpy as np
import copy
import itertools
import cv2

def constant_augmentation(img, tag):
    # tag: 1:水平翻转, 2:垂直翻转 , 3-9:旋转 per 45°, 10-16:水平+旋转, 0: img itself
    if tag == 0:
        return img

    size = img.shape  # 获得图像的形状
    h = size[0]
    w = size[1]

    if tag == 1:  # 水平翻转
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i, j in itertools.product(range(h), range(w)):
            iLR[i, w - 1 - j] = img[i, j]
        return iLR
    if tag == 2:  # 垂直翻转
        jLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i, j in itertools.product(range(w), range(h)):
            jLR[h - 1 - j, i] = img[j, i]
        return jLR
    if tag <= 9 and tag >= 3:  # 旋转
        angle = (tag - 2) * 45
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv2.warpAffine(img, M, (newW, newH))  # borderValue 缺省，默认是黑色
    if tag <= 16 and tag >= 10:  # 水平+旋转
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(h):  # 元素循环
            for j in range(w):
                iLR[i, w - 1 - j] = img[i, j]

        angle = (tag - 9) * 45
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv2.warpAffine(iLR, M, (newW, newH))  # borderValue 缺省，默认是黑色
    print('error in data_augmentation')
    return None  # error



# path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-1/'
path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-2/'

file = 'fold1_train.txt'

''' 1. 处理为变换数据，使其和变换后的组织形式一样
import numpy as np
path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-1/'

with open(path+'fold1_train.txt', "r") as f:
    train_list = f.read()
train_list = train_list.split('\n')

with open(path+'fold1_test.txt', "r") as f:
    test_list = f.read()
test_list = test_list.split('\n')

data_list = train_list + test_list

for name in data_list:
    d = np.load(path+name, allow_pickle=True)
    image1, image2, label = d[0].astype('float32'), d[1].astype('float32'), d[3].astype('float32')
    pcr, prompt = d[4], d[2][np.newaxis,:].astype('float32')
    npy = [image1[np.newaxis,:], image2[np.newaxis,:], label[np.newaxis,:], pcr, prompt]
    np.save(f"{path}{name}", np.array(npy, dtype=object))

'''

''' 2. 测试集也增广（已经是新的组织形式）
path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-1/'
file = 'fold1_test.txt'

with open(path+file, "r") as f:
    train_list = f.read()
train_list = train_list.split('\n')

tags = [1,2]
tt=0
for name in train_list:
    d = np.load(path+name, allow_pickle=True)
    image1, image2, label = d[0][0], d[1][0], d[2][0]
    pcr, prompt = d[3], d[4]
    for tag in tags:
        image1_, image2_, label_ = [], [], []
        for i, j, k in zip(image1, image2, label):
            image1_.append(constant_augmentation(i,tag))
            image2_.append(constant_augmentation(j,tag))
            label_.append(constant_augmentation(k,tag))
        image1_ = np.array(image1_)
        image2_ = np.array(image2_)
        label_ = np.array(label_)
        if tt<6:
            tt+=1
            print(np.unique(label_))
        npy = [image1_[np.newaxis,:], image2_[np.newaxis,:], label_[np.newaxis,:], pcr, prompt]
        name_ = f'{name[:-4]}_{tag}.npy'
        np.save(f"{path}{name_}", np.array(npy, dtype=object))


# 不需要保存列表

'''

''' 3. 其它折的训练集也生成增广的列表
path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-3/'

# tags = [1,2]
tags = [1,2,4,6,8,11,15]
for i in range(1, 6):
    list_add = []
    # with open(path+f'fold{i}_train.txt', "r") as f:
    with open(path+f'fold{i}_test.txt', "r") as f:
        train_list = f.read()
    train_list = train_list.split('\n')
    print(i, len(train_list))
    for name in train_list:
        for tag in tags:
            name_ = f'{name[:-4]}_{tag}.npy'
            list_add.append(name_)
    
    str = '\n'
    print(i, len(train_list)+len(list_add))
    # with open(f"{path}/fold{i}_train_aug12.txt","w") as f:
    # with open(f"{path}/fold{i}_test_aug12.txt","w") as f:
    with open(f"{path}/fold{i}_test_aug7times.txt","w") as f:
        f.write(str.join(train_list + list_add))

'''


with open(path+file, "r") as f:
    train_list = f.read()
train_list = train_list.split('\n')

tags = [1,2]
list_add = []
tt=0
for name in train_list:
    d = np.load(path+name, allow_pickle=True)
    image1, image2, label = d[0].astype('float32'), d[1].astype('float32'), d[3].astype('float32')
    pcr, prompt = d[4], d[2][np.newaxis,:].astype('float32')
    for tag in tags:
        image1_, image2_, label_ = [], [], []
        for i, j, k in zip(image1, image2, label):
            image1_.append(constant_augmentation(i,tag))
            image2_.append(constant_augmentation(j,tag))
            label_.append(constant_augmentation(k,tag))
        image1_ = np.array(image1_)
        image2_ = np.array(image2_)
        label_ = np.array(label_)
        if tt<6:
            tt+=1
            print(np.unique(label_))
        npy = [image1_[np.newaxis,:], image2_[np.newaxis,:], label_[np.newaxis,:], pcr, prompt]
        name_ = f'{name[:-4]}_{tag}.npy'
        np.save(f"{path}{name_}", np.array(npy, dtype=object))
        list_add.append(name_)

str = '\n'
with open(f"{path}/fold1_train_aug12.txt","w") as f:
    f.write(str.join(train_list + list_add))
