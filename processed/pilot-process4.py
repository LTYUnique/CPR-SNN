'''
与process3相比，去掉两个与prompt强相关的信息
'''
import os
import pandas as pd
import SimpleITK as sitk
from sklearn.preprocessing import scale
import numpy as np
import skimage.transform as trans

import copy
import itertools
import cv2

def constant_augmentation(img, tag):
    # TODO: 2和13重了
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


def norm(img):  # norm to [0,1]
    win_min = img.min()
    win_max = img.max()
    img = img if (win_max == win_min) else (img - win_min) / (win_max - win_min)
    # return img.astype(np.uint8)
    return img.astype('float32')


# 读取表
xls_file = '/root/ffy/data/breast_data/SharedClinicalAndRFS.xls'

df = pd.read_excel(xls_file, sheet_name=1, usecols=['Patient ID', 'MRI 3', 'Clinical response']) # 读取部分列
# 去NAN
df = df.dropna(axis=0,how='any')
assert df.shape[0] == 51
# 编码临床信息

df_prompt = pd.read_excel(xls_file, sheet_name=1, usecols=['Patient ID', 'SER Volume 1 (cc)', 'SER Volume 3 (cc)', 'chemo', 'AC only=0, taxol=1', 'Clinical size pre', 'Clinical size post', 'Cancer type', 'Surgery type', 'path size (cm)', 'LN', 'volume_1', 'volume_3']) # 读取部分列
# 只用'volume_1'和'volume_3'是因为这两列数据更全
df_prompt = pd.get_dummies(df_prompt,columns=["chemo"], drop_first=True) # 离散
df_prompt = pd.get_dummies(df_prompt,columns=["AC only=0, taxol=1"], drop_first=True)
df_prompt = pd.get_dummies(df_prompt,columns=["Cancer type"], drop_first=True)
df_prompt = pd.get_dummies(df_prompt,columns=["Surgery type"], drop_first=True)
df_prompt = pd.get_dummies(df_prompt,columns=["LN"], drop_first=True)

df_prompt["SER Volume 1 (cc)"] = scale(df_prompt["SER Volume 1 (cc)"]) # 连续
df_prompt["SER Volume 3 (cc)"] = scale(df_prompt["SER Volume 3 (cc)"])
df_prompt["Clinical size pre"] = scale(df_prompt["Clinical size pre"])
df_prompt["Clinical size post"] = scale(df_prompt["Clinical size post"])
df_prompt["path size (cm)"] = scale(df_prompt["path size (cm)"])
df_prompt["volume_1"] = scale(df_prompt["volume_1"])
df_prompt["volume_3"] = scale(df_prompt["volume_3"])

df_prompt = df_prompt.set_index('Patient ID')
df_prompt = df_prompt.fillna(0)
print(df_prompt.shape)
# 29列

path = '/root/ffy/data/breast_data/processed/' # from
save_path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-5'
reshape = (64, 128, 128)
# save_path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-5'
# reshape = (64, 256, 256)

os.makedirs(save_path, exist_ok=True)
name_list = []
tags = [1,2,4,6,8,11,15]
# 遍历df
for i, row in df.iterrows():
    index = row["Patient ID"]
    sub_path = path+index.replace('_', '-')+'/MRI3'  # processed/UCSF-BR-01/MRI3
    sub_list = os.listdir(sub_path)
    assert 'T1.nii' in sub_list and 'T2.nii' in sub_list, index
    # continue
    # use T1, T2, seg1
    T1 = sitk.GetArrayFromImage(sitk.ReadImage(sub_path+'/T1.nii'))
    T2 = sitk.GetArrayFromImage(sitk.ReadImage(sub_path+'/T2.nii'))
    label = sitk.GetArrayFromImage(sitk.ReadImage(sub_path+'/seg1.nii'))
    assert np.max(label) == 1
    # img process  row: 256 256 60 or 512 512 64
    T1 = trans.resize(T1, reshape)
    T1 = norm(T1)
    T2 = trans.resize(T2, reshape)
    T2 = norm(T2)
    label_resize = trans.resize(label, reshape)
    assert np.max(label_resize)<=1 and np.min(label_resize)==0, (np.max(label_resize), np.min(label_resize))
    label =  np.zeros_like(label_resize)
    label[label_resize>0.5] = 1
    label = label.astype('float32')
    # prompt
    prompt = np.array(df_prompt.loc[index])
    # pcr
    pcr = row['Clinical response'] # shape: (29,)
    pcr = 0 if pcr > 1 else 1
    # print(index, pcr)

    # save
    npy = np.array([T1[np.newaxis,:], T2[np.newaxis,:], label[np.newaxis,:], pcr, prompt[np.newaxis,:]])
    np.save(f"{save_path}/{index}.npy", npy)
    name_list.append(index+'.npy')
    # aug
    for tag in tags:
        image1_, image2_, label_ = [], [], []
        for t1, t2, k in zip(T1, T2, label):
            image1_.append(constant_augmentation(t1,tag))
            image2_.append(constant_augmentation(t2,tag))
            label_.append(constant_augmentation(k,tag))
        image1_ = np.array(image1_)
        image2_ = np.array(image2_)
        label_ = np.array(label_)
        npy = np.array([image1_[np.newaxis,:], image2_[np.newaxis,:], label_[np.newaxis,:], pcr, prompt[np.newaxis,:]])
        np.save(f"{save_path}/{index}_{tag}.npy", npy)

# txt同pilot3

'''
def resample_3D_nii_to_Fixed_size(nii_image, image_new_size, resample_methold=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    image_original_size = nii_image.GetSize()  # 原始图像的尺寸
    image_original_spacing = nii_image.GetSpacing()  # 原始图像的像素之间的距离
    image_new_size = np.array(image_new_size, float)
    factor = image_original_size / image_new_size
    image_new_spacing = image_original_spacing * factor
    image_new_size = image_new_size.astype(np.int)

    resampler.SetReferenceImage(nii_image)  # 需要resize的图像（原始图像）
    resampler.SetSize(image_new_size.tolist())
    resampler.SetOutputSpacing(image_new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_methold)

    return resampler.Execute(nii_image)

sub_path = '/root/ffy/data/breast_data/processed/UCSF-BR-68/MRI3'  # processed/UCSF-BR-01/MRI3
label = sitk.ReadImage(sub_path+'/seg1.nii')
label_2 = resample_3D_nii_to_Fixed_size(label, (512, 512, 64), resample_methold=sitk.sitkNearestNeighbor)

label = sitk.GetArrayFromImage(label)
label_resize = trans.resize(label, reshape)
label_1 =  np.zeros_like(label_resize)
label_1[label_resize>0.5] = 1

sitk.WriteImage(sitk.GetImageFromArray(label_1), '63-label1.nii')
sitk.WriteImage(label_2, '63-label2_.nii')
sitk.WriteImage(sitk.GetImageFromArray(label), '63-label.nii')

两种resize方法其实是差不多的
emmmm...其实还是resample_3D_nii那个好一些吧

和原图shape相差并不大的话，二者都能保留细节

reshape
label = sitk.ReadImage(sub_path+'/seg1.nii')
label = sitk.GetArrayFromImage(label)
label.shape
label_resize = trans.resize(label, reshape)
label_1 =  np.zeros_like(label_resize)
label_1[label_resize>0.5] = 1
sitk.WriteImage(sitk.GetImageFromArray(label_1), '63-label1_.nii')
'''