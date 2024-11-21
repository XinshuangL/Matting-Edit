import cv2
import numpy as np
import torch
from torch.nn import functional as F
import sys
import json
import networks
from   utils import CONFIG
import utils
import toml
import os
import tqdm
import copy

fg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
fg_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

def pred2alpha(pred):
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
    mask = pred['mask']
    
    alpha_pred = mask.clone().detach()

    weight_os8 = utils.get_unknown_tensor(mask)
    weight_os8[...] = 1

    weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
    weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
    
    alpha_pred[weight_os8>0] = alpha_pred_os8[weight_os8>0]
    alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
    alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

    return alpha_pred

def get_color_loss(fg1_pred, fg2_pred, fg1, fg2, alpha1, alpha2, bg_pred, bg):
    dif1_AD = (fg1_pred - fg1).abs() * alpha1
    dif2_AD = (fg2_pred - fg2).abs() * alpha2
    difbg_AD = (bg_pred - bg).abs()

    dif1_SE = (fg1_pred - fg1) * (fg1_pred - fg1) * alpha1
    dif2_SE = (fg2_pred - fg2) * (fg2_pred - fg2) * alpha2
    difbg_SE = (bg_pred - bg) * (bg_pred - bg)

    MAD = (dif1_AD.sum() / alpha1.sum() + dif2_AD.sum() / alpha2.sum()) / 2
    MSE = (dif1_SE.sum() / alpha1.sum() + dif2_SE.sum() / alpha2.sum()) / 2

    MAD_bg = difbg_AD.mean()
    MSE_bg = difbg_SE.mean()

    return (MAD + MAD_bg) / 2, (MSE + MSE_bg) / 2

def alpha2box(alpha):
    alpha_single = copy.deepcopy(alpha)
    alpha_single[alpha_single>127] = 255
    alpha_single[alpha_single<=127] = 0

    fg_set = np.where(alpha_single != 0)
    x_min = np.min(fg_set[1])
    x_max = np.max(fg_set[1])
    y_min = np.min(fg_set[0])
    y_max = np.max(fg_set[0])
    return np.array([x_min, y_min, x_max, y_max])

def undo_norm(fg):
    return fg * fg_std + fg_mean

def RGB2np(img):
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = img*255
    img[img>255] = 255
    img[img<0] = 0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img

def alpha2np(alpha, binary=False):
    alpha = alpha[0].detach().cpu().numpy()
    alpha = alpha*255
    alpha[alpha>255] = 255
    alpha[alpha<0] = 0
    if binary:
        alpha[alpha > 1] = 255
    return alpha.astype(np.uint8)
