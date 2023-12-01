import os
import torch

import sys
sys.path.append('models')
from models.model import Model7
from torch.utils.data import DataLoader
from src.data import TestGen_pilot
import numpy as np
from sklearn.metrics import accuracy_score

basic_channel = 64
num_classes = 2
batch_size = 2
x1_scale = 1
x2_scale = 1
model_path = '/root/ffy/Ltyv/multi_task/log/multi_pilot_13f'

model = Model7(basic_channel, num_classes, (128,128,64), batch_size, x1_scale, x2_scale, 29)
device = 'cpu'
# device = 'cuda'

data_path = '/root/ffy/Ltyv/multi_task/processed/pilot/pilot-5'
def load_test(fold):
    # get ids 
    # with open(f"{data_path}/fold{fold}_test.txt", "r") as f:  #打开文本  #, encoding='utf-8'
    with open(f"{data_path}/fold{fold}_test_someaug.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        test_list = f.read()   #读取文本
    test_list = test_list.split('\n')

    test_set = TestGen_pilot('', data_path, test_list, False, 29)#TestGen('', data_path, test_list)

    loader_args = dict(batch_size=batch_size, num_workers=1)
    return DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

for fold in range(1, 6):
        use_model = f'{model_path}{fold}/best-model.pth'
        model.load_state_dict(torch.load(use_model, map_location=torch.device(device)))
        model.to(device)

        pre = []
        gt = []

        test_loader = load_test(fold)
        model.eval()
        for names, image1, image2, label, pcr, prompt in test_loader:
                image1 = image1.to(device)
                image2 = image2.to(device)
                prompt = prompt.to(device)
                img_out, pcr_pre = model.predict(image1, image2, prompt)
                pre.extend(pcr_pre.cpu().tolist())
                gt.extend(pcr.cpu().tolist())
                # save
                img_out = img_out.cpu().numpy()
                label = label.numpy()

        pre = np.array(pre)
        
        path = model_path.replace('log/', 'predict/') + str(fold)
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/pre_someaug.npy", pre)
        np.save(f"{path}/gt_someaug.npy", np.array(gt))
        # np.save(f"{path}/pre.npy", pre)
        # np.save(f"{path}/gt.npy", np.array(gt))
        
        pre_label = np.argmax(pre, axis=1)
        print('fold', fold, accuracy_score(gt, pre_label))