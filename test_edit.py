import tqdm
import cv2
import numpy as np
import json
import os

def add2dict(d, k, v):
    if k in d.keys():
        d[k] += v
    else:
        d[k] = v
    return d

def dict_div(d, count):
    for k in d.keys():
        d[k] = float(d[k] / count)
    return d

def merge2(alpha1, alpha2, fg1, fg2, bg):
    alpha2_occlude = alpha2 * (1 - alpha1)
    alpha_bg = (1 - alpha1) * (1 - alpha2)
    return fg1 * alpha1 + fg2 * alpha2_occlude + bg * alpha_bg

def check_result(pred, gt):
    MAD = np.abs(pred - gt).mean()
    MSE = ((pred - gt)*(pred - gt)).mean()

    return MSE, MAD

for dataset_name in ['Distinctions-646', 'AMD']:
    for gt_box in [False, True]:

        data_root = f'../{dataset_name}/generated/'
        pred_root = f'ours_{dataset_name}_{gt_box}/'

        results = {}
        count = 0

        for i in tqdm.tqdm(range(1000)):
            image_path = f'{data_root}/{i}/composite.png'
            bg_path = f'{data_root}/{i}/bg.png'
            fg1_path = f'{data_root}/{i}/fg1.png'
            fg2_path = f'{data_root}/{i}/fg2.png'
            alpha1_path = f'{data_root}/{i}/alpha1.png'
            alpha2_path = f'{data_root}/{i}/alpha2.png'

            fg1_pred_path = f'{pred_root}/{i}/fg1_refine.png'
            fg2_pred_path = f'{pred_root}/{i}/fg2_refine.png'
            bg_pred_path = f'{pred_root}/{i}/bg_refine.png'
            alpha1_pred_path = f'{pred_root}/{i}/alpha1.png'
            alpha2_pred_path = f'{pred_root}/{i}/alpha2.png'

            bg = cv2.imread(bg_path).astype(np.float32) / 255
            fg1 = cv2.imread(fg1_path).astype(np.float32) / 255
            fg2 = cv2.imread(fg2_path).astype(np.float32) / 255
            alpha1 = cv2.imread(alpha1_path).astype(np.float32) / 255
            alpha2 = cv2.imread(alpha2_path).astype(np.float32) / 255

            bg_pred = cv2.imread(bg_pred_path).astype(np.float32) / 255
            fg1_pred = cv2.imread(fg1_pred_path).astype(np.float32) / 255
            fg2_pred = cv2.imread(fg2_pred_path).astype(np.float32) / 255
            alpha1_pred = cv2.imread(alpha1_pred_path).astype(np.float32) / 255
            alpha2_pred = cv2.imread(alpha2_pred_path).astype(np.float32) / 255

            # remove
            remove_gt1 = fg1 * alpha1 + bg * (1 - alpha1)
            remove_gt2 = fg2 * alpha2 + bg * (1 - alpha2)
            remove_pred1 = fg1_pred * alpha1_pred + bg_pred * (1 - alpha1_pred)
            remove_pred2 = fg2_pred * alpha2_pred + bg_pred * (1 - alpha2_pred)
            remove_MSE1, remove_MAD1 = check_result(remove_pred1, remove_gt1)
            remove_MSE2, remove_MAD2 = check_result(remove_pred2, remove_gt2)
            remove_MSE = (remove_MSE1 + remove_MSE2) / 2
            remove_MAD = (remove_MAD1 + remove_MAD2) / 2
            results = add2dict(results, 'remove_MSE', remove_MSE)
            results = add2dict(results, 'remove_MAD', remove_MAD)

            # swap
            swap_gt = merge2(alpha2, alpha1, fg2, fg1, bg)
            swap_pred = merge2(alpha2_pred, alpha1_pred, fg2_pred, fg1_pred, bg_pred)
            swap_MSE, swap_MAD = check_result(swap_pred, remove_gt1)
            results = add2dict(results, 'swap_MSE', swap_MSE)
            results = add2dict(results, 'swap_MAD', swap_MAD)

            # mean
            results = add2dict(results, 'mean_MSE', (remove_MSE + swap_MSE) / 2)
            results = add2dict(results, 'mean_MAD', (remove_MAD + swap_MAD) / 2)

            count += 1

            os.makedirs(f'edit_ours/results/ours_{dataset_name}_{gt_box}/{i}', exist_ok=True)
            cv2.imwrite(f'edit_ours/results/ours_{dataset_name}_{gt_box}/{i}/remove_pred1.png', remove_pred1*255)
            cv2.imwrite(f'edit_ours/results/ours_{dataset_name}_{gt_box}/{i}/remove_pred2.png', remove_pred2*255)
            cv2.imwrite(f'edit_ours/results/ours_{dataset_name}_{gt_box}/{i}/swap_pred.png', swap_pred*255)

        results = dict_div(results, count)

        with open(f'edit_ours/edit_ours_{dataset_name}_{gt_box}.json', 'w') as f:
            json.dump(results, f)
