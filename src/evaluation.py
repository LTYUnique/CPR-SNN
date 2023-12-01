import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from utils import resample_3D_nii_to_Fixed_size
from metrics import Metirc

def convert_probability_to_mask_array(predict, choice):
    predict = sitk.GetArrayFromImage(predict)
    mask = np.zeros_like(predict, dtype='uint8')
    mask[predict > choice] = 1
    return mask



class Evaluation():
    def __init__(self, save_path, choice=0, metrics='all', Global=False):
        # 增加指标只需修改3处：value of metrics, result and fun
        '''
        Global: True, False, only
        '''
        if metrics == 'all':
            self.metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR']  # 'RVD', 
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
        self.Global = Global
        self.Single = False if Global=='only' else True
        self.PRE = []
        self.GT = []
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        self.choice = choice

    
    def cacu(self, name, predict, groundtruth):        
        pre2 = np.zeros_like(predict, dtype='uint8')
        pre2[predict > self.choice] = 1
        gt2 = np.zeros_like(groundtruth, dtype='uint8')
        gt2[groundtruth > 0] = 1
        
        if self.Global:
            self.PRE.append(pre2)
            self.GT.append(gt2)
        if self.Single:
            self.result['name'].append(name)
            metirc = Metirc(pre2, gt2)  # 得有
            for m in self.metrics:
                self.result[m].append(eval(self.fun[m]))

    def view(self, get_dice=None):
        dice=None
        if self.Single:
            print('Single:')
            for m in self.metrics:
                print(m, ':', np.mean(self.result[m]))
            if get_dice=="Single": dice=np.mean(self.result['dice'])
        if self.Global:
            print('Global:')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                print(m, eval(self.fun[m]))
            if get_dice=='Global': dice=metirc.dice_coef()
        return dice
    
    def get_dice(self, form):
        '''
        form (str): 'Single' or 'Global'.
        '''
        if form == 'Single' and self.Single:
            print('Single:')
            return np.mean(self.result['dice'])  # 即使已经添加了dice均值也不影响
        if form == 'Global' and self.Global:
            print('Global:')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            return metirc.dice_coef()
        raise Exception(f'error combination. form:{form}, Single{self.Single}, Global:{self.Global}')
            
    def save(self, csv_name):
        if self.Single:
            print('Single:')
            self.result['name'].append('means')
            for m in self.metrics:
                mean = np.mean(self.result[m])
                print(m, ':', mean)
                self.result[m].append(mean)
        if self.Global:
            print(f'Global:({np.array(self.PRE).shape})({np.array(self.GT).shape})')
            self.result['name'].append('Global')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                g = eval(self.fun[m])
                print(m, g)
                self.result[m].append(g)
        df = pd.DataFrame(self.result)
        df.to_csv(self.save_path + '/' + csv_name, index=False)
        # print(df)







def evaluation(save_path, pred_dir, gt_dir, choice=0, resize=None, metrics='all'):
    # 增加指标只需修改3处：value of metrics, result and fun
    if metrics == 'all':
        metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR']  # 'RVD', 
    result={
        'name':[],
        'dice':[],
        'Jaccard':[],
        'recall':[],
        'precision':[],
        # 'RVD':[],
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



def connected_domain_2(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = list(range(1, num_label+1))
    area_list = [stats.GetNumberOfPixels(l) for l in range(1, num_label +1)]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for i in num_list_sorted[1:]:
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask

    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    return output

def evaluation_connected_domain_2(save_path, pred_dir, gt_dir, choice=0, resize=None, metrics='all'):
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
        sitk.WriteImage(sitk.GetImageFromArray(predict), f'connected_domain/pre_{name}')
        predict = connected_domain_2(predict) if np.max(predict)>0 else predict
        #predict = sitk.GetArrayFromImage(predict)
        sitk.WriteImage(sitk.GetImageFromArray(predict), f'connected_domain/cd_{name}')
        # print(predict.shape)

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


if __name__ == '__main__':
    '''
    evaluation_connected_domain_2(
        save_path='/root/ffy/diffusion/my_diffusion/csv_save/ISPY1_train01/epoch-300-model_connected_domain.csv',
        pred_dir='/root/ffy/diffusion/my_diffusion/predict3D/ISPY1_train01/epoch-300-model',
        gt_dir='/root/ffy/data/processed/ISPY1/label3D',
        choice=0,
        resize=128,
        metrics='all')
    '''
    
    evaluation_connected_domain_2(
        save_path='/root/ffy/diffusion/my_diffusion/csv_save/abdomen_train01/epoch-300-model_connected_domain.csv',
        pred_dir='/root/ffy/diffusion/my_diffusion/predict3D/abdomen_train01/epoch-300-model',
        gt_dir='/root/ffy/data/processed/abdomen/label_6_3D',
        choice=0,
        resize=128,
        metrics='all')

    