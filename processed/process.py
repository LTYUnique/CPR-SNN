'''
3D
将数据提前处理好，存成npy格式以加快训练速度
'''

import os
import numpy as np
import skimage.transform as trans
import SimpleITK as sitk

import warnings
warnings.filterwarnings('ignore')

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

def resize(image1, image2, label, size, z_size=48):
    '''input : Image'''
    image1 = sitk.GetArrayFromImage(image1)
    image1 = trans.resize(image1, (z_size, size, size))

    image2 = sitk.GetArrayFromImage(image2)
    image2 = trans.resize(image2, (z_size, size, size))

    label = resample_3D_nii_to_Fixed_size(label, (size, size, z_size), resample_methold=sitk.sitkNearestNeighbor)
    return image1, image2, sitk.GetArrayFromImage(label)

def norm(img):  # norm to [0,1]
    win_min = img.min()
    win_max = img.max()
    img = img if (win_max == win_min) else (img - win_min) / (win_max - win_min)
    # return img.astype(np.uint8)
    return img.astype('float32')


# sourcery skip: avoid-builtin-shadow
from_file = '/root/ffy/breast_data/processed'
to_file = 'processed/breast'


file_list = os.listdir(from_file)  # 64个病人
os.makedirs(f'{to_file}/MRI1_T1_T2', exist_ok=True)

for file_name in file_list:
    # image
    img1 = sitk.ReadImage(f'{from_file}/{file_name}/MRI1/T1.nii')
    img2 = sitk.ReadImage(f'{from_file}/{file_name}/MRI1/T2.nii')
    print(img1.GetSize(), end=' -> ')
    # label
    seg = sitk.ReadImage(f'{from_file}/{file_name}/MRI1/seg1.nii')
    # resize
    img1, img2, seg = resize(img1, img2, seg, 128)
    # norm
    img1 = norm(img1)
    img2 = norm(img2)
    seg = norm(seg)
    #save
    npy = [img1, img2, seg]
    np.save(f"{to_file}/MRI1_T1_T2/{file_name}.npy", npy)

val_index = int(len(file_list)*0.7)
str = '\n'
with open(f"{to_file}/MRI1_T1_T2/train_list.txt","w") as f:
    f.write(str.join(file_list[:val_index]))
with open(f"{to_file}/MRI1_T1_T2/val_list.txt","w") as f:
    f.write(str.join(file_list[val_index]))
with open(f"{to_file}/MRI1_T1_T2/test_list.txt","w") as f:
    f.write(str.join(file_list[val_index+1:]))



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
