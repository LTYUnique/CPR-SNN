import torch
import skimage.transform as trans
from scipy import ndimage
import numpy as np

import SimpleITK as sitk
import cv2
import random
import copy
import argparse

def train_args(parser):
    epochs = 200
    epoch_start = 1  # index
    # checkpoint_rate = 10 # 每隔多少个epoch记录下结果
    
    # save
    description = 'train01'
    val_dir = 'val'
    log_dir = 'log'  # save pth for every epoch
    
    
    loss = 'BCE'
    
    
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--epoch_start', default=epoch_start, type=int)
    # parser.add_argument('--checkpoint_rate', default=checkpoint_rate, type=int)
    parser.add_argument('--checkpoint_rate_pth', default=20, type=int)
    parser.add_argument('--checkpoint_rate_val', default=20, type=int)

    parser.add_argument('--description', '-name', default=description, type=str)
    parser.add_argument('--val_dir', default=val_dir, type=str)
    parser.add_argument('--log_dir', default=log_dir, type=str)
    
    
    parser.add_argument('--loss', default=loss, type=str)
    parser.add_argument('--loss_coefficient', default=0.5, type=float) # 分类+分割时，分割损失的系数
    parser.add_argument('--dice_smooth', default=1.0, type=float)
    parser.add_argument('--seg_loss_reduce', default=1.0, type=float) # only for seg_loss
    return parser
	
def test_args(parser):

    # DSC
    choice = 0.5
    # n_classes = 2

    # test
    parser.add_argument('--use_model', required=True, type=str, help='log/train06/epoch-320-model.pth')
    parser.add_argument('--label', required=True, type=str, help='label1')
    parser.add_argument('--choice', default=choice, type=int)
    # parser.add_argument('--n_classes', default=n_classes, type=int)


    
    return parser

def set_args(tag):
    # model
    batch_size=1
    num_workers = 1
    img_size = 128
    lr = 2e-4  # learning_rate
    basic_channel = 32
    
    # model = 'model1'
    
    # model1
    layer_num = 5  # 多少层网络
    # model2+
    x1_scale = 1
    x2_scale = 1

    # data
    # resize = 128
    # p = 0.3  # 多少用于测试

    parser = argparse.ArgumentParser(description='initial')

    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    # parser.add_argument('--use_sigmoid', action='store_true', help='use sigmoid after resnet18')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--basic_channel', default=basic_channel, type=int)
    parser.add_argument('--layer_num', default=layer_num, type=int)
    parser.add_argument('--batch_size', '-bs', default=batch_size, type=int)
    parser.add_argument('--img_size', default=img_size, type=int)
    # parser.add_argument('--resize', default=resize, type=int)
    parser.add_argument('--num_workers', default=num_workers, type=int)

    parser.add_argument('--x1_scale', default=x1_scale, type=float)
    parser.add_argument('--x2_scale', default=x2_scale, type=float)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--num_classes', default=2, type=int)

    parser.add_argument('--root', default='')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--train_list', default='train_list', type=str)
    parser.add_argument('--test_list', default='test_list', type=str)

    parser.add_argument('--data', required=True, type=str, help='default should be ISPY1')
    parser.add_argument('--n_clin_var', required=True, type=int, help='size of prompt, 8 in ISPY1')
    parser.add_argument('--data_deep', type=int, default=96, help='D')

    parser.add_argument('--zero_prompt_test', action='store_true', help='change prompt to all zero on test')

    
    # train
    if 'train' in tag:
        parser = train_args(parser)

    # test
    if 'test' in tag:
        parser = test_args(parser)

    return parser.parse_args()




def read_pics(*paths):
    pics = []
    for path in paths:
        if path.endswith('.nii') or path.endswith('.dcm') or path.endswith('.nii.gz'):
            pic = sitk.ReadImage(path)
            pic = sitk.GetArrayFromImage(pic)
        else:
            pic = cv2.imread(path, 0)
        pics.append(pic)
    return pics if len(pics) > 1 else pics[0]


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


def nii_resize_2D(image, label, shape):
    """
    type of image,label: Image or array or None

    :return: array or None
    """
    # image
    if isinstance(image, sitk.SimpleITK.Image):  # image need type array, if not, transform it
        image = sitk.GetArrayFromImage(image)
    if image is not None:
        image = trans.resize(image, (shape, shape))
    # label
    if isinstance(label, np.ndarray):
        label = sitk.GetImageFromArray(label)  # label1 need type Image
    if label is not None:
        label = resample_3D_nii_to_Fixed_size(label, (shape, shape),
                                              resample_methold=sitk.sitkNearestNeighbor)
        label = sitk.GetArrayFromImage(label)
    return image, label


def norm(img):  # norm to [0,1]
    win_min = img.min()
    win_max = img.max()
    img = img if (win_max == win_min) else (img - win_min) / (win_max - win_min)
    # return img.astype(np.uint8)
    return img.astype('float32')

def random_rot_flip(image_1, image_2, label):
    k = np.random.randint(0, 4)
    image_1 = np.rot90(image_1, k)
    image_2 = np.rot90(image_2, k)
    axis = np.random.randint(0, 2)
    image_1 = np.flip(image_1, axis=axis).copy()
    image_2 = np.flip(image_2, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    return image_1, image_2, label


def random_rotate(image_1, image_2, label):
    angle = np.random.randint(-20, 20)
    image_1 = ndimage.rotate(image_1, angle, order=0, reshape=False)
    image_2 = ndimage.rotate(image_2, angle, order=0, reshape=False)
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image_1, image_2, label

def data_augmentation(image_1, image_2, label, tag):
    if tag == 0:
        return image_1, image_2, label
    
    elif tag == 1:
        if random.random() > 0.5:
            return random_rot_flip(image_1, image_2, label)  # image, label
        if random.random() > 0.5:
            return random_rotate(image_1, image_2, label)
        return image_1, image_2, label

    else:
        return constant_augmentation(image_1, eval(tag)),  constant_augmentation(image_2, eval(tag)),  constant_augmentation(label, eval(tag))

'''
tags_value = {  # 0: '',  # 原始图像
    1: 'a',  # 水平翻转
    2: 'b',  # 垂直翻转
    3: 'c',  # 旋转45°
    4: 'd',  # 旋转90°
    5: 'e',  # 旋转135°
    6: 'f',  # 旋转180°
    7: 'g',  # 旋转225°
    8: 'h',  # 旋转270°
    9: 'i',  # 旋转315°
    10: 'j',  # 水平翻转+旋转45°
    11: 'k',  # 水平翻转+旋转90°
    12: 'l',  # 水平翻转+旋转135°
    13: 'm',  # 水平翻转+旋转180°
    14: 'n',  # 水平翻转+旋转225°
    15: 'o',  # 水平翻转+旋转270°
    16: 'p'  # 水平翻转+旋转315°
}

'''
def constant_augmentation(img, tag):
    # tag: 1:水平翻转, 2:垂直翻转 , 3-9:旋转 per 45°, 10-16:水平+旋转, 0: img itself
    if tag == 0:
        return img

    size = img.shape  # 获得图像的形状
    h = size[0]
    w = size[1]

    if tag == 1:  # 水平翻转
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(h):  # 元素循环
            for j in range(w):
                iLR[i, w - 1 - j] = img[i, j]
        return iLR
    if tag == 2:  # 垂直翻转
        jLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(w):  # 元素循环
            for j in range(h):
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

if __name__ == '__main__':
    a = np.random.randn(5,5)
    print(norm(a))