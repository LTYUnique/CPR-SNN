import os

import natsort
import numpy as np
import SimpleITK as sitk

from utils import read_pics
from tqdm import tqdm

import pandas as pd

'''
Lits: only index(named as ID)
abdomen: the same

breast: have ID like UCSF-BR-**
image and label hava the same tag
'''

def adapt_convert(file_dir, csv_info, p):
    df = pd.read_csv(csv_info, index_col=[0], usecols=['index', 'ID'])
    # print(df.head())
    # exit()
    id_list = list(set(df.index.tolist()))
    val_index = int(len(id_list) * (1-p))  # also length of train
    test_list = df.loc[(df.index > val_index)]['ID']
    test_list = list(set(test_list.tolist()))
    # print(test_list)
    
    save_path = file_dir.replace('predict', 'predict3D')
    # print(save_path)
    os.makedirs(save_path, exist_ok=True)
    ids = natsort.natsorted(os.listdir(file_dir))
    
    for id in test_list:
        ids_sub = list(filter(lambda x: x.startswith(id), ids))
        image_3D = []
        for i in range(len(ids_sub)):
            name = f'{id}_{i}.nii'
            image_2D = read_pics(f'{file_dir}/{name}')
            image_3D.append(image_2D)
        image_3D = np.array(image_3D)
        sitk.WriteImage(sitk.GetImageFromArray(image_3D), f'{save_path}/{id}.nii')

        
    # print(ids_sub)
    return save_path


def convert(file_dir, start_index_or_csv_info, end_index_or_p, tag='Lits'):  # image_index doesn't include 'end_index'
    if tag == 'Lits':
        start = 'volume-'
        label_start = 'segmentation-'
    elif tag == 'abdomen':
        start = 'img00'
        label_start = 'label00'
    elif tag in ['breast', 'ISPY1']:
        csv_info = start_index_or_csv_info
        p = end_index_or_p
        return adapt_convert(file_dir, csv_info, p)
    else:
        raise RuntimeError('error data name', tag)
        
    start_index = start_index_or_csv_info
    end_index = end_index_or_p

    save_path = file_dir.replace('predict', 'predict3D')
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    ids = natsort.natsorted(os.listdir(file_dir))
    # 原始图像的序号范围
    index_range = range(start_index, end_index)

    with tqdm(total=end_index-start_index, desc='convert', unit='img') as pbar:
        for i, image_index in enumerate(index_range):
            # 提取
            ids_sub = list(filter(lambda x: x.startswith(f'{start}{image_index}'), ids))
            #print(len(ids_sub), f'{start}{image_index}')
            # 合并
            image_3D = []
            for image_name in ids_sub:
                image_2D = read_pics(f'{file_dir}/{image_name}')
                image_3D.append(image_2D)
            image_3D = np.array(image_3D)

            # 存储
            sitk.WriteImage(sitk.GetImageFromArray(image_3D), f'{save_path}/{label_start}{image_index}.nii')
            pbar.update()  # 更新进度
    return save_path

if __name__ == '__main__':
    convert(
        file_dir='predict/ISPY1_train01/epoch-300-model',
        start_index_or_csv_info='/root/ffy/data/processed/ISPY1/info_32.csv', end_index_or_p=0.3,
        tag='ISPY1'
    )
