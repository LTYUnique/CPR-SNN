#!/root/anaconda3/bin/python

import os
import sys

# from tqdm import tqdm;
sys.path.append('models')
sys.path.append('src')

import torch
from torch.optim import Adam
import torch.nn.functional as F 

from models.model import *
from src.data import load_train, load_test
from src.utils import set_args
# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
# import SimpleITK as sitk
# import cv2
import skimage.transform as trans

import numpy as np
from sklearn.metrics import *
from src.evaluation import Evaluation


def get_model(args):
    if args.model == 'model2':
        model = Model2(args.basic_channel, args.num_classes, args.x1_scale, args.x2_scale)
    elif args.model == 'model3':
        model = Model3(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model4':
        model = Model4(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model5':
        model = Model5(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model6':
        model = Model6(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model7':
        model = Model7(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model8':
        model = Model8(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    elif args.model == 'model9':
        model = Model9(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
    else:
        Model = eval(args.model)
        model = Model(args.basic_channel, args.num_classes, (128,128,args.data_deep), args.batch_size, 
                       args.x1_scale, args.x2_scale, args.n_clin_var)
        # raise RuntimeError('error model name', args.model)
    return model
'''
class Loss_func():
    def __init__(self, loss):
        super(Loss_func, self).__init__()
        self.BCELoss = torch.nn.BCELoss()
        weights = torch.Tensor([46/156.0, 111/156.0])
        self.CELoss = torch.nn.CrossEntropyLoss(weight=weights)
        if loss == 'multi_loss':
            self.get_loss = self.loss_func3
        else:
            raise RuntimeError('error loss type:', loss)
    
    def loss_func3(self, u4, u3, u2, img_out, mask, y_pre, y_true):
        loss1 = self.BCELoss(img_out, mask)
        loss2 = self.dice_coeff(img_out, mask)
        loss3 = self.CELoss(y_pre, y_true.long())

        # b, c, h, w = mask.shape
        mask_down = trans.resize(mask.cpu().numpy(), u2.shape)
        loss_u2 = self.BCELoss(u2, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u3.shape)
        loss_u3 = self.BCELoss(u3, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u4.shape)
        loss_u4 = self.BCELoss(u4, torch.from_numpy(mask_down).cuda())

        return (loss1+loss2 + loss3+(loss_u2+loss_u3+loss_u4)/3)*0.5

    def dice_coeff(self, pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
 
        return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def __call__(self, u4, u3, u2, img_out, mask, y_pre, y_true):
        return self.get_loss(u4, u3, u2, img_out, mask, y_pre, y_true)
'''

class Loss_func():
    def __init__(self, args):
        super(Loss_func, self).__init__()
        self.BCELoss1 = torch.nn.BCELoss()
        self.BCELoss2 = torch.nn.BCELoss()  # 这几个是后来加的
        self.BCELoss3 = torch.nn.BCELoss()
        self.BCELoss4 = torch.nn.BCELoss()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.l=args.loss_coefficient
        self.smooth = args.dice_smooth
        if args.loss == 'multi_loss':
            self.get_loss = self.loss_func3
        elif args.loss == 'seg_loss':
            self.get_loss = self.loss_func4
            self.seg_loss_reduce = args.seg_loss_reduce
        elif args.loss == 'seg_loss2':
            self.get_loss = self.loss_func5
            self.seg_loss_reduce = args.seg_loss_reduce
        elif args.loss == 'class_loss':
            self.get_loss = self.loss_func6
        else:
            raise RuntimeError('error loss type:', args.loss)
    
    def loss_func3(self, u4, u3, u2, img_out, mask, y_pre, y_true):
        loss1 = self.BCELoss1(img_out, mask)
        loss2 = self.dice_coeff(img_out, mask)
        loss3 = self.CELoss(y_pre, y_true.long())

        # b, c, h, w = mask.shape
        mask_down = trans.resize(mask.cpu().numpy(), u2.shape)
        loss_u2 = self.BCELoss2(u2, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u3.shape)
        loss_u3 = self.BCELoss3(u3, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u4.shape)
        loss_u4 = self.BCELoss4(u4, torch.from_numpy(mask_down).cuda())

        return (loss1+loss2 + (loss_u2+loss_u3+loss_u4)/3)*self.l + loss3*(1-self.l)

    def loss_func4(self, u4, u3, u2, img_out, mask, y_pre=None, y_true=None):  # seg only
        loss1 = self.BCELoss1(img_out, mask)
        loss2 = self.dice_coeff(img_out, mask)

        # b, c, h, w = mask.shape
        mask_down = trans.resize(mask.cpu().numpy(), u2.shape)
        loss_u2 = self.BCELoss2(u2, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u3.shape)
        loss_u3 = self.BCELoss3(u3, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u4.shape)
        loss_u4 = self.BCELoss4(u4, torch.from_numpy(mask_down).cuda())

        return (loss1+loss2 + (loss_u2+loss_u3+loss_u4)/3) *self.seg_loss_reduce

    def loss_func5(self, u4, u3, u2, img_out, mask, y_pre=None, y_true=None):
        # loss4的基础上去掉一个深度监督
        loss1 = self.BCELoss1(img_out, mask)
        loss2 = self.dice_coeff(img_out, mask)

        # b, c, h, w = mask.shape
        mask_down = trans.resize(mask.cpu().numpy(), u2.shape)
        loss_u2 = self.BCELoss2(u2, torch.from_numpy(mask_down).cuda())

        mask_down = trans.resize(mask_down, u3.shape)
        loss_u3 = self.BCELoss3(u3, torch.from_numpy(mask_down).cuda())

        # mask_down = trans.resize(mask_down, u4.shape)
        # loss_u4 = self.BCELoss4(u4, torch.from_numpy(mask_down).cuda())

        return (loss1+loss2 + (loss_u2+loss_u3)/2) *self.seg_loss_reduce

    def dice_coeff(self, pred, target):
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
 
        return 1-(2. * intersection + self.smooth) / (m1.sum() + m2.sum() + self.smooth)

    def loss_func6(self, u4, u3, u2, img_out, mask, y_pre, y_true):
        return self.CELoss(y_pre, y_true.long())

    def __call__(self, u4, u3, u2, img_out, mask, y_pre, y_true):
        return self.get_loss(u4, u3, u2, img_out, mask, y_pre, y_true)


def train(args):
    # model initial
    model = get_model(args)
    model.to(args.device)    
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # print('load:', args.load)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=args.device))
        optimizer.load_state_dict(torch.load(args.load.replace('-model', '-optim')))
    
    # print(next(model.parameters()).device)
    loss_func = Loss_func(args)
    if args.model == 'model9':
        test_func = test2
    elif args.model == 'Ablation3_1_noSeg':
        test_func = test3
    else: test_func = test
    
    # dataset
    train_loader = load_train(args)
    test_loader = load_test(args)

    writer = SummaryWriter(log_dir=f"runs/{args.description}")
    # print('make dir:', f"{args.log_dir}/{args.description}")
    os.makedirs(f"{args.log_dir}/{args.description}", exist_ok=True)
    best_acc, best_epoch = 0, None
    for epoch in range(args.epoch_start, args.epochs+1):
        # print('train star')
        # exit()
        model.train()
        total_loss = 0
        # with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
        for idx, (image_T1, image_T2, mask, pcr, prompt) in enumerate(train_loader):
            image_T1 = image_T1.to(args.device)
            image_T2 = image_T2.to(args.device)
            mask = mask.to(args.device)
            pcr = pcr.to(args.device)
            prompt = prompt.to(args.device)
            # print(image_T1.dtype, image_T2.dtype, mask.dtype, prompt.dtype)  # all float32
            # exit()
            u4, u3, u2, img_out, pcr_pre = model(image_T1, image_T2, prompt)
            loss = loss_func(u4, u3, u2, img_out, mask, pcr_pre, pcr)
            # print(loss.item())
            # print('--------------***--------')
            writer.add_scalar(f'epoch_loss/{epoch}', loss.item(), idx)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # break

            # pbar.update()  # 更新进度

        ave_loss = total_loss/len(train_loader)
        writer.add_scalar('all_epoch/loss', ave_loss, epoch)
        writer.flush()
        # test for every epoch
        # if epoch > 80:
        acc, cap = test_func(model, test_loader, args.device)
        print(f'epoch {epoch}, loss=', ave_loss, datetime.datetime.now(), cap)
        print('---------------------------------------------')
        # exit()
        if acc > best_acc and cap:
            best_acc = acc
            best_epoch = epoch
            # 保存模型
            model_filename = f"{args.log_dir}/{args.description}/best-model.pth"
            optim_filename = f"{args.log_dir}/{args.description}/best-optim.pth"
            torch.save(model.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)
        # if epoch % args.checkpoint_rate_pth == 0:
        #     model_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-model.pth"
        #     optim_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-optim.pth"
        #     torch.save(model.state_dict(), model_filename)
        #     torch.save(optimizer.state_dict(), optim_filename)
        # # n个epoch后，采样看看效果
        # if epoch % args.checkpoint_rate_val == 0:
        #     os.makedirs(f"{args.val_dir}/{args.description}/{epoch}", exist_ok=True)
        #     model.eval()
        #     with torch.no_grad():
        #         for names, image1, image2 in val_loader:
        #             image1 = image1.to(args.device)
        #             image2 = image2.to(args.device)
        #             #print('----------------------', image.shape)
        #             u4, u3, u2, img_out = model(image1, image2)
        #             # save
        #             img_out = img_out.cpu().numpy()                    
        #             for mask, name in zip(img_out, names):
        #                 mask = mask.squeeze()  # 1*128*128 ->128*128
        #                 #sitk.WriteImage(sitk.GetImageFromArray(mask), f"{args.val_dir}/{args.description}/{epoch}/{name[:-4]}.nii")
        #                 cv2.imwrite(f"{args.val_dir}/{args.description}/{epoch}/{name[:-4]}.png", mask*255)  # *255

    # if epoch % args.checkpoint_rate_pth != 0:  # 模型尚未保存
    #     model_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-model.pth"
    #     optim_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-optim.pth"
    #     torch.save(model.state_dict(), model_filename)
    #     torch.save(optimizer.state_dict(), optim_filename)

    print(f'best result: epoch{best_epoch}:{best_acc}')

    return model

@torch.no_grad()
def test(model, test_loader, device):
    evaluation = Evaluation(None, choice=0.5)
    model.eval()
    pre = []
    gt = []
    name_list = []
    # with tqdm(total=len(test_loader), desc='predict', unit='img') as pbar:
    for names, image1, image2, label, pcr, prompt in test_loader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        prompt = prompt.to(device)
        img_out, pcr_pre = model.predict(image1, image2, prompt)
        pre.extend(pcr_pre.cpu().tolist())
        gt.extend(pcr.cpu().tolist())
        name_list.extend(list(names))
        # save
        img_out = img_out.cpu().numpy()
        label = label.numpy()
        for mask, label_, name in zip(img_out, label, names):
            mask = mask.squeeze()  # 1*128*128 ->128*128
            evaluation.cacu(name, mask, label_)

            # pbar.update()  # 更新进度
    evaluation.view()

    pre = np.array(pre)
    pre_label = np.argmax(pre, axis=1)
    confusion = confusion_matrix(gt, pre_label)
    print(confusion)
    capable = True if (confusion[0][0] and confusion[1][1]) else False
    acc = accuracy_score(gt, pre_label)
    print('accuracy', acc)
    print('roc_auc', roc_auc_score(gt, pre_label))
    if capable:
        print('precision', precision_score(gt, pre_label))
        print('recall', recall_score(gt, pre_label))
        print('f1_score', f1_score(gt, pre_label, pos_label= 1, average='binary'))

    # print(capable)
    """ # 仅pilot数据集且增广时适用
    selected_o_gt = [gt[i] for i, name in enumerate(name_list) if len(name) == 14]
    selected_o_pre = [pre_label[i] for i, name in enumerate(name_list) if len(name) == 14]

    # selected_pre = [pre_label[i] for i, name in enumerate(name_list) if '_1.npy' in name or '_2.npy' in name]
    # selected_gt = [gt[i] for i, name in enumerate(name_list) if '_1.npy' in name or '_2.npy' in name]

    print('-----------------------')
    print(f'no aug:\n {confusion_matrix(selected_o_gt, selected_o_pre)}')
    print(accuracy_score(selected_o_gt, selected_o_pre), precision_score(selected_o_gt, selected_o_pre), recall_score(selected_o_gt, selected_o_pre), roc_auc_score(selected_o_gt, selected_o_pre), f1_score(selected_o_gt, selected_o_pre))
    """
    '''
    selected_pre = selected_pre + selected_o_pre
    selected_gt = selected_gt + selected_o_gt
    print('aug12:\n', confusion_matrix(selected_gt, selected_pre))
    print(accuracy_score(selected_gt, selected_pre), precision_score(selected_gt, selected_pre), recall_score(selected_gt, selected_pre), roc_auc_score(selected_gt, selected_pre), f1_score(selected_gt, selected_pre))
    print('---------------------------------------------')
    '''
    return acc, capable


@torch.no_grad()
def test2(model, test_loader, device):
    evaluation = Evaluation(None, choice=0.5)
    model.eval()
    # with tqdm(total=len(test_loader), desc='predict', unit='img') as pbar:
    for names, image1, image2, label, pcr, prompt in test_loader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        prompt = prompt.to(device)
        img_out = model.predict(image1, image2, prompt)
        img_out = img_out.cpu().numpy()
        label = label.numpy()
        for mask, label_, name in zip(img_out, label, names):
            mask = mask.squeeze()  # 1*128*128 ->128*128
            evaluation.cacu(name, mask, label_)

            # pbar.update()  # 更新进度
    return evaluation.view(get_dice='Single'), True


@torch.no_grad()
def test3(model, test_loader, device):
    model.eval()
    pre = []
    gt = []
    name_list = []
    # with tqdm(total=len(test_loader), desc='predict', unit='img') as pbar:
    for names, image1, image2, label, pcr, prompt in test_loader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        prompt = prompt.to(device)
        pcr_pre = model.predict(image1, image2, prompt)
        pre.extend(pcr_pre.cpu().tolist())
        gt.extend(pcr.cpu().tolist())
        name_list.extend(list(names))
        
    pre = np.array(pre)
    pre_label = np.argmax(pre, axis=1)
    confusion = confusion_matrix(gt, pre_label)
    print(confusion)
    capable = True if (confusion[0][0] and confusion[1][1]) else False
    acc = accuracy_score(gt, pre_label)
    print('accuracy', acc)
    print('roc_auc', roc_auc_score(gt, pre_label))
    if capable:
        print('precision', precision_score(gt, pre_label))
        print('recall', recall_score(gt, pre_label))
        print('f1_score', f1_score(gt, pre_label, pos_label= 1, average='binary'))
    # print(capable)
    
    '''
    selected_o_gt = [gt[i] for i, name in enumerate(name_list) if len(name) == 14]
    selected_o_pre = [pre_label[i] for i, name in enumerate(name_list) if len(name) == 14]

    # selected_pre = [pre_label[i] for i, name in enumerate(name_list) if '_1.npy' in name or '_2.npy' in name]
    # selected_gt = [gt[i] for i, name in enumerate(name_list) if '_1.npy' in name or '_2.npy' in name]
    print('-----------------------')
    print(f'no aug:\n {confusion_matrix(selected_o_gt, selected_o_pre)}')
    print(accuracy_score(selected_o_gt, selected_o_pre), precision_score(selected_o_gt, selected_o_pre), recall_score(selected_o_gt, selected_o_pre), roc_auc_score(selected_o_gt, selected_o_pre), f1_score(selected_o_gt, selected_o_pre))
    '''
    return acc, capable


def setting(tag):
    args = set_args(tag=tag)

    print(args)
    # assert args.loss == 'multi_loss'
    #exit()
    if args.load:
        assert args.epoch_start > 1, 'you have used a pre trained model, reset epoch_start please'
    else: assert args.epoch_start == 1, 'epoch_start should be zero'

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    args.device = device
    return args


if __name__ == '__main__':
    args = setting(tag=['train'])
    # model = train(args)
    train(args)
