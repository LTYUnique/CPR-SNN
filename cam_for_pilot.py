import os
import torch
import numpy as np
import argparse
import cv2
import skimage.transform as trans
import SimpleITK as sitk
# import os
import sys;
sys.path.append('models')
from models.model import Model7
from src.data import load_test

parser = argparse.ArgumentParser(description='initial')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--tag', type=str, required=True, help='model name')
parser.add_argument('--i', type=int, default=1)
parser.add_argument('--file', type=str, default='test_list.txt')
args = parser.parse_args()

heatmap_weight = 0.4
'''
for pilot:

CUDA_VISIBLE_DEVICES=1 python cam_for_pilot.py --model_path=log/multi_pilot_13f1/best-model.pth --file=fold3_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f1

nohup python cam_for_pilot.py --model_path=log/multi_pilot_13f1/best-model.pth --file=fold1_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f1 &&

nohup python cam_for_pilot.py --model_path=log/multi_pilot_13f2/best-model.pth --file=fold2_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f2 &&

nohup python cam_for_pilot.py --model_path=log/multi_pilot_13f3/best-model.pth --file=fold3_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f3 &&

nohup python cam_for_pilot.py --model_path=log/multi_pilot_13f4/best-model.pth --file=fold4_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f4 &&

nohup python cam_for_pilot.py --model_path=log/multi_pilot_13f5/best-model.pth --file=fold5_test.txt --data_path=processed/pilot/pilot-5/ --tag=pilot13_f5 &

'''

def get_model_pilot():
    basic_channel = 64
    num_classes = 2
    batch_size = 2
    x1_scale = 1
    x2_scale = 1
    return Model7(basic_channel, num_classes, (128,128,64), batch_size, x1_scale, x2_scale, 29)

model_path = args.model_path
model = get_model_pilot()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.to('cuda')
model.eval()

tag = args.tag
# device = 'cuda'

args.root = ''
args.data = 'pilot'
args.test_list = args.file[:-4]
args.zero_prompt_test=False
args.n_clin_var = 29
args.batch_size = 2
args.num_workers = 0
test_loader = load_test(args)
for names, image1, image2, label, pcr, prompt in test_loader:
    # image1 = image1.to(device)
    # image2 = image2.to(device)
    # prompt = prompt.to(device)
    predictions, activations = model.cam_pcr(image1, image2, prompt)
    # ---------------------------------------------------------
    predictions[0][int(pcr[0])].backward() # args.i
    weights = np.mean(model.gradients.detach().cpu().numpy(), axis=(2, 3, 4))[0, :]
    # 使用权重对激活层进行加权求和
    cam = np.sum(weights[:, np.newaxis, np.newaxis, np.newaxis] * activations[-1].detach().cpu().numpy(), axis=0)
    cam = np.maximum(cam, 0)
    if cam.max == 0:
        print('max of cam is zero')
    else:
        cam = cam / cam.max()

    # 将 CAM 可视化到原始图像上
    # print(cam.shape) # (6, 8, 8)
    cam = trans.resize(cam, (96, 128, 128))
    save_path = f'cam/{tag}/gt{pcr[0]}/{names[0]}_{round(float(predictions[0][0]),2)}_{round(float(predictions[0][1]),2)}'
    os.makedirs(save_path, exist_ok=True)
    # print(image1[0].shape) # [1, 64, 128, 128]
    # exit()
    for i, (cam_slice, img_slice) in enumerate(zip(cam, image1[0][0])):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_slice), cv2.COLORMAP_JET)
        # print('heatmap', heatmap.shape)
        img_slice = np.uint8(255 * img_slice.cpu())
        # print('img_slice', img_slice.shape)
        img_slice = np.broadcast_to(img_slice[:,:, np.newaxis],(128,128,3))
        # print(heatmap.shape, img.shape, np.max(heatmap), np.min(heatmap))
        result = heatmap * heatmap_weight + img_slice*(1-heatmap_weight)

        # name_ = name[:-4]
        # image
        cv2.imwrite(f'{save_path}/{i+1}_img.jpg', img_slice)
        cv2.imwrite(f"{save_path}/{i+1}.jpg", result)
    # exit()

    # s = np.array([d[0][0],d[1][0]])
    sitk.WriteImage(sitk.GetImageFromArray(label[0]), f'{save_path}/seg.nii')
    # ---------------------------------------------------------
    predictions, activations = model.cam_pcr(image1, image2, prompt)
    predictions[1][int(pcr[1])].backward() # args.i
    weights = np.mean(model.gradients.detach().cpu().numpy(), axis=(2, 3, 4))[0, :]
    # 使用权重对激活层进行加权求和
    cam = np.sum(weights[:, np.newaxis, np.newaxis, np.newaxis] * activations[-1].detach().cpu().numpy(), axis=0)
    cam = np.maximum(cam, 0)
    if cam.max == 0:
        print('max of cam is zero')
    else:
        cam = cam / cam.max()

    # 将 CAM 可视化到原始图像上
    # print(cam.shape) # (6, 8, 8)
    cam = trans.resize(cam, (96, 128, 128))
    save_path = f'cam/{tag}/gt{pcr[1]}/{names[1]}_{round(float(predictions[1][0]),2)}_{round(float(predictions[1][1]),2)}'
    os.makedirs(save_path, exist_ok=True)
    # print(image1[0].shape) # [1, 64, 128, 128]
    # exit()
    for i, (cam_slice, img_slice) in enumerate(zip(cam, image1[1][0])):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_slice), cv2.COLORMAP_JET)
        # print('heatmap', heatmap.shape)
        img_slice = np.uint8(255 * img_slice.cpu())
        # print('img_slice', img_slice.shape)
        img_slice = np.broadcast_to(img_slice[:,:, np.newaxis],(128,128,3))
        # print(heatmap.shape, img.shape, np.max(heatmap), np.min(heatmap))
        result = heatmap * heatmap_weight + img_slice*(1-heatmap_weight)

        # name_ = name[:-4]
        # image
        cv2.imwrite(f'{save_path}/{i+1}_img.jpg', img_slice)
        cv2.imwrite(f"{save_path}/{i+1}.jpg", result)
    # exit()

    # s = np.array([d[0][0],d[1][0]])
    sitk.WriteImage(sitk.GetImageFromArray(label[1]), f'{save_path}/seg.nii')
    # exit()


# # os.makedirs(f'cam/{tag}', exist_ok=True)
# with open(f"{args.data_path}/{args.file}", "r") as f:  #打开文本  #, encoding='utf-8'
#     img_list = f.read()   #读取文本
# img_list = img_list.split('\n')

# print('total of image:', len(img_list))
# for name in img_list:
#     d = np.load(args.data_path+name, allow_pickle=True)
#     # if d[3]:
#     img1, img2, prompt = d[0], d[1], d[4]
#     # print('d0', d[0].shape)
#     # for i in d[0]:
#     #     print(i.shape)
#     #     exit()
#     img1 = torch.tensor(img1[np.newaxis,:], requires_grad=True)
#     img2 = torch.tensor(img2[np.newaxis,:], requires_grad=True)
#     prompt = torch.tensor(prompt[np.newaxis,:], requires_grad=True)
#     # 使用 Grad-CAM 类得到激活层和预测结果
#     # grad_cam = GradCAM(model=model)  #, candidate_layers=['layers'] 
#     # activations, predictions = grad_cam(img)
#     # print(img1.shape)
#     img1 = torch.cat((img1, img1), dim=0)
#     img2 = torch.cat((img2, img2), dim=0)
#     prompt = torch.cat((prompt, prompt), dim=0)
#     # print(img1.shape)
#     # exit()
#     predictions, activations = model.cam_pcr(img1, img2, prompt)
#     # print(activations.shape) # [1, 1024, 6, 8, 8]

#     # 获取梯度并计算权重
#     predictions[0][int(d[3])].backward() # args.i
#     # weights = np.mean(grad_cam.gradients.detach().cpu().numpy(), axis=(2, 3))[0, :]
#     weights = np.mean(model.gradients.detach().cpu().numpy(), axis=(2, 3, 4))[0, :]
#     # 使用权重对激活层进行加权求和
#     cam = np.sum(weights[:, np.newaxis, np.newaxis, np.newaxis] * activations[-1].detach().cpu().numpy(), axis=0)
#     cam = np.maximum(cam, 0)
#     if cam.max == 0:
#         print('max of cam is zero')
#     else:
#         cam = cam / cam.max()

#     # 将 CAM 可视化到原始图像上
#     # print(cam.shape) # (6, 8, 8)
#     cam = trans.resize(cam, (96, 128, 128))
#     save_path = f'cam/{tag}/gt{d[3]}/{name}_{round(float(predictions[0][0]),2)}_{round(float(predictions[0][1]),2)}'
#     os.makedirs(save_path, exist_ok=True)
#     # exit()
#     for i, (cam_slice, img_slice) in enumerate(zip(cam, d[0][0])):
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_slice), cv2.COLORMAP_JET)
#         # print(heatmap.shape)
#         img_slice = np.uint8(255 * img_slice)
#         # print(img_slice.shape)
#         img_slice = np.broadcast_to(img_slice[:,:, np.newaxis],(128,128,3))
#         # print(heatmap.shape, img.shape, np.max(heatmap), np.min(heatmap))
#         result = heatmap * heatmap_weight + img_slice*(1-heatmap_weight)

#         # name_ = name[:-4]
#         # image
#         cv2.imwrite(f'{save_path}/{i+1}_img.jpg', img_slice)
#         cv2.imwrite(f"{save_path}/{i+1}.jpg", result)
#     # exit()

#     # s = np.array([d[0][0],d[1][0]])
#     sitk.WriteImage(sitk.GetImageFromArray(d[2][0]), f'{save_path}/seg.nii')