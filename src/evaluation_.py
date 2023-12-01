import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from utils import resample_3D_nii_to_Fixed_size
from metrics import Metirc


class Evaluation():
    def __init__(self, save_path, choice=0, metrics='all'):
        # 增加指标只需修改3处：value of metrics, result and fun
        if metrics == 'all':
            self.metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR']  #  'RVD',
        else:
            self.metrics = metrics

        self.result={
            'name':[],
            'dice':[],
            'Jaccard':[],
            'recall':[],
            'precision':[],
            # 'RVD':[],
            'FNR':[],
            'FPR':[]
        }
        self.fun = {
            'dice':'metirc.dice_coef()',
            'Jaccard':'metirc.iou_score()',
            'recall':'metirc.recall()',
            'precision':'metirc.precision()',
            'RVD':'metirc.RVD()',
            'FNR':'metirc.FNR()',
            'FPR':'metirc.FPR()'
        }
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        self.choice = choice
    
    def convert_probability_to_mask_array(self, predict, choice):
        mask = np.zeros_like(predict, dtype='uint8')
        mask[predict > choice] = 1
        return mask
    
    def cacu(self, name, predict, groundtruth):
        predict = self.convert_probability_to_mask_array(predict, self.choice)
        self.result['name'].append(name)
        
        metirc = Metirc(predict, groundtruth)
        for m in self.metrics:
            self.result[m].append(eval(self.fun[m]))

    def view(self):
        for m in self.metrics:
            print(m, ':', np.mean(self.result[m]))
        
    def save(self, csv_name):
        self.result['name'].append('means')
        for m in self.metrics:
            self.result[m].append(np.mean(self.result[m]))

        df = pd.DataFrame(self.result)
        df.to_csv(self.save_path + '/' + csv_name, index=False)
        print(df)

'''
def evaluation(save_path, pred_dir, gt_dir, choice=0, resize=None, metrics='all'):
    # 增加指标只需修改3处：value of metrics, result and fun
    if metrics == 'all':
        metrics = ['dice', 'Jaccard', 'recall', 'precision', 'RVD', 'FNR', 'FPR']
    result={
        'name':[],
        'dice':[],
        'Jaccard':[],
        'recall':[],
        'precision':[],
        'RVD':[],
        'FNR':[],
        'FPR':[]
    }
    fun = {
        'dice':'metirc.dice_coef()',
        'Jaccard':'metirc.iou_score()',
        'recall':'metirc.recall()',
        'precision':'metirc.precision()',
        'RVD':'metirc.RVD()',
        'FNR':'metirc.FNR()',
        'FPR':'metirc.FPR()'
    }
    os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)
    pred_filenames = os.listdir(pred_dir)

    for i in range(len(pred_filenames)):
        name = pred_filenames[i]

        predict = sitk.ReadImage(os.path.join(pred_dir, name))
        predict = convert_probability_to_mask_array(predict, choice)
        #predict = sitk.GetArrayFromImage(predict)

        groundtruth = sitk.ReadImage(os.path.join(gt_dir, name))
        if resize:
            groundtruth = resample_3D_nii_to_Fixed_size(groundtruth, (resize, resize,
                            groundtruth.GetSize()[2]),
                            resample_methold=sitk.sitkNearestNeighbor)
        groundtruth = sitk.GetArrayFromImage(groundtruth)

        result['name'].append(name)
        
        metirc = Metirc(predict, groundtruth)
        for m in metrics:
            result[m].append(eval(fun[m]))

    # 计算均值 todo: All arrays must be of the same length
    result['name'].append('means')
    for m in metrics:
        result[m].append(np.mean(result[m]))

    df = pd.DataFrame(result)
    df.to_csv(save_path, index=False)
    print(df)
'''
