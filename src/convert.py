import os

import natsort
import numpy as np
import SimpleITK as sitk

from utils import read_pics
from tqdm import tqdm

def convert(file_dir, start_index, end_index, tag='Lits'):  # image_index doesn't include 'end_index'
	if tag == 'Lits':
		start = 'volume-'
		label_start = 'segmentation-'
	elif tag == 'abdomen':
		start = 'img00'
		label_start = 'label00'
	elif tag == 'breast':
		start = 'UCSF-BR-'
		label_start = 'UCSF-BR-'
	else:
		raise RuntimeError('error data name', tag)
		
	save_path = file_dir.replace('predict', 'predict3D')
	print('convert to:', save_path)
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
		file_dir='predict/train09/epoch-330-model',
		start_index=105, end_index=131
	)
