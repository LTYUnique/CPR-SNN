import os
import sys;
sys.path.append('models')
sys.path.append('src')

import torch

from models.model import *
from src.data import load_test
from src.utils import set_args
from sklearn.metrics import *
from src.evaluation import Evaluation

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def get_model(args):
    if args.model == 'model2':
        model = Model2(args.basic_channel,args.num_classes, args.x1_scale, args.x2_scale)
    elif args.model == 'model3':
        model = Model3(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.x1_scale, args.x2_scale)
    elif args.model == 'model4':
        model = Model4(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale)
    elif args.model == 'model5':
        model = Model5(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale)
    elif args.model == 'model6':
        model = Model6(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale)
    elif args.model == 'model7':
        model = Model7(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model8':
        model = Model8(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale)
    else:
        Model = eval(args.model)
        model = Model(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale)
    return model
   
@torch.no_grad()
def predict(args, model):
    if model is None:
        model = get_model(args)
        model.load_state_dict(torch.load(args.use_model, map_location=torch.device(args.device)))
        model.to(args.device)
        path = f"{args.use_model[:-4].replace('log/', 'predict/')}"
    else:
        model_path = f"{args.log_dir}/{args.description}/epoch-{args.epochs}-model.pth"
        path = f"predict/{model_path[:-4].replace('log/', '')}"
    suffix = f'_{args.label}' if args.label else ''
    test_loader = load_test(args)

    print('predict to:', path)
    os.makedirs(path, exist_ok=True)
    evaluation = Evaluation(path, choice=args.choice)
    model.eval()
    pre = []
    gt = []
    with tqdm(total=len(test_loader), desc='predict', unit='img') as pbar:
        for names, image1, image2, label, pcr, prompt in test_loader:
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            prompt = prompt.to(args.device)
            img_out, pcr_pre = model.predict(image1, image2, prompt)
            pre.extend(pcr_pre.cpu().tolist())
            gt.extend(pcr.cpu().tolist())
            # save
            img_out = img_out.cpu().numpy()
            label = label.numpy()
            for mask, label_, name in zip(img_out, label, names):
                mask = mask.squeeze()  # 1*128*128 ->128*128
                # sitk.WriteImage(sitk.GetImageFromArray(mask), f'{path}/{name[:-4]}.nii')
                evaluation.cacu(name, mask, label_)

            pbar.update()  # 更新进度
    evaluation.save(f'result{suffix}.csv')

    pre = np.array(pre)
    np.save(f"{path}/pre.npy", pre)
    np.save(f"{path}/gt.npy", np.array(gt))

    pre_label = np.argmax(pre, axis=1)
    print(confusion_matrix(gt, pre_label))
    print('accuracy', accuracy_score(gt, pre_label))
    print('precision', precision_score(gt, pre_label))
    print('recall', recall_score(gt, pre_label))
    print('roc_auc', roc_auc_score(gt, pre_label))
    print('f1_score', f1_score(gt, pre_label, pos_label= 1, average='binary'))
    
    # print(confusion_matrix(pre_label, gt))
    # print('accuracy', accuracy_score( pre_label, gt))
    # print('precision', precision_score(pre_label, gt))
    # print('recall', recall_score( pre_label, gt))
    # print('roc_auc', roc_auc_score(pre_label, gt))
    # print


def setting(tag):
    args = set_args(tag=tag)
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    args.device = device
    print(args)
    return args


if __name__ == '__main__':
    args = setting(tag=['test'])
    # model = test(args)
    predict(args, model=None)

'''
python test.py --model=model7  --data_path='processed/pilot/pilot-1' --test_list=fold1_test --basic_channel=64 --use_model=log/multi_ISPY1_m7_03_fold0/best-model.pth --num_workers=0 --batch_size=1 --label='' --data=pilot --zero_prompt_test --n_clin_var=8

python test.py --model=model7  --data_path='processed/pilot/pilot-1' --test_list=fold5_test_aug12 --basic_channel=64 --use_model=log/multi_pilot_03f5/best-model.pth --num_workers=0 --batch_size=1 --label='' --data=pilot --n_clin_var=2

python test.py --model=model7  --data_path='processed/pilot/pilot-3' --data_deep=64 --test_list=fold1_test_aug7times --basic_channel=64 --use_model=log/multi_pilot_07/best-model.pth --num_workers=0 --batch_size=2 --label='' --data=pilot --n_clin_var=2






CUDA_VISIBLE_DEVICES=1 python test.py --model=model7  --data_path='processed/pilot/pilot-4' --data_deep=64 --test_list=fold1_test_someaug --basic_channel=64 --use_model=log/multi_pilot_12f1/best-model.pth --num_workers=1 --batch_size=2 --label='' --data=pilot --n_clin_var=31

CUDA_VISIBLE_DEVICES=1 python test.py --model=model7  --data_path='processed/pilot/pilot-4' --data_deep=64 --test_list=fold2_test_someaug --basic_channel=64 --use_model=log/multi_pilot_12f2/best-model.pth --num_workers=1 --batch_size=2 --label='' --data=pilot --n_clin_var=31

CUDA_VISIBLE_DEVICES=1 python test.py --model=model7  --data_path='processed/pilot/pilot-4' --data_deep=64 --test_list=fold3_test_someaug --basic_channel=64 --use_model=log/multi_pilot_12f3/best-model.pth --num_workers=1 --batch_size=2 --label='' --data=pilot --n_clin_var=31

CUDA_VISIBLE_DEVICES=1 python test.py --model=model7  --data_path='processed/pilot/pilot-4' --data_deep=64 --test_list=fold4_test_someaug --basic_channel=64 --use_model=log/multi_pilot_12f4/best-model.pth --num_workers=1 --batch_size=2 --label='' --data=pilot --n_clin_var=31

CUDA_VISIBLE_DEVICES=1 python test.py --model=model7  --data_path='processed/pilot/pilot-4' --data_deep=64 --test_list=fold5_test_someaug --basic_channel=64 --use_model=log/multi_pilot_12f5/best-model.pth --num_workers=1 --batch_size=2 --label='' --data=pilot --n_clin_var=31

'''