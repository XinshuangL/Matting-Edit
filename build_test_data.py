dataset_name = 'Distinctions-646' # Please select Distinctions-646 or AMD
bg_root = '../BG-20k/testval/'
bg_shift = 0

H, W = 1024, 1024

import os
import glob

import PIL.Image
from reproducibility import *
import cv2
import PIL
import json
import numpy as np
import tqdm

from ground_dino import *

name_dict_path = f'image_names/name_dict_{dataset_name}.json'
fg_root = f'../{dataset_name}/test/fg/'
alpha_root = f'../{dataset_name}/test/alpha/'
out_root = f'../{dataset_name}/generated/'

with open(name_dict_path, 'r') as f:
    name_dict = json.load(f)

bg_paths = glob.glob(bg_root + '/*')
fg_paths = glob.glob(fg_root + '/*')

fg_paths.sort()
bg_paths.sort()

names = [path.split('/')[-1].split('\\')[-1].split('.')[0] for path in fg_paths]

data = {}
for name in names:
    fg = cv2.imread(f'{fg_root}/{name}.png').astype(np.float32) / 255.
    alpha = cv2.imread(f'{alpha_root}/{name}.png').astype(np.float32) / 255.
    category = name_dict[name].replace('.', '').replace(' ', '')
    data[name] = {
        'fg': fg, 'alpha': alpha, 'category': category
    }

def shrink(fg, alpha, mode):
    H, W, _ = fg.shape
    if random.randint(0, 1) == 0:
        H_len = round(H * 0.75)
        W_len = round(W * 0.75)
    else:
        H_len = round(H * 2/3)
        W_len = round(W * 2/3)
    
    if mode in [0, 1]:
        H_low = 0
        H_high = H_len
    else:
        H_low = H - H_len
        H_high = H

    if mode in [0, 2]:
        W_low = 0
        W_high = W_len
    else:
        W_low = W - W_len
        W_high = W
        
    new_fg = np.zeros_like(fg, dtype=np.float32)
    new_alpha = np.zeros_like(alpha, dtype=np.float32)

    fg = cv2.resize(fg, (W_len, H_len))
    alpha = cv2.resize(alpha, (W_len, H_len))

    new_fg[H_low:H_high,W_low:W_high] = fg
    new_alpha[H_low:H_high,W_low:W_high] = alpha

    return new_fg, new_alpha

cur_id = 0

log = {}

for name in tqdm.tqdm(names):
    cur_data = data[name]
    fg = cur_data['fg']
    alpha = cur_data['alpha']
    category = cur_data['category']

    fg_real = cv2.resize(fg, (W, H))
    alpha_real = cv2.resize(alpha, (W, H))

    for _ in range(20):
        fg = fg_real
        alpha = alpha_real

        ok = False
        while not ok:
            other_name = random.choice(names)
            other_category = data[other_name]['category']
            if not other_category == category:
                ok = True
        other_data = data[other_name]
        other_fg = other_data['fg']
        other_alpha = other_data['alpha']
        other_category = other_data['category']

        bg = cv2.imread(bg_paths[cur_id + bg_shift]).astype(np.float32) / 255.

        other_fg = cv2.resize(other_fg, (W, H))
        other_alpha = cv2.resize(other_alpha, (W, H))
        bg = cv2.resize(bg, (W, H))

        mode1, mode2 = random.sample([0, 1, 2, 3], 2)
        fg, alpha = shrink(fg, alpha, mode1)
        other_fg, other_alpha = shrink(other_fg, other_alpha, mode2)

        other_alpha_occlude = other_alpha * (1 - alpha)
        alpha_bg = (1 - alpha) * (1 - other_alpha)
        composite = fg * alpha + other_fg * other_alpha_occlude + bg * alpha_bg
        
        cur_out_root = f'{out_root}/{cur_id}/'
        os.makedirs(cur_out_root, exist_ok=True)
        
        cv2.imwrite(f'{cur_out_root}/composite.png', composite*255)
        cv2.imwrite(f'{cur_out_root}/fg1.png', fg*255)
        cv2.imwrite(f'{cur_out_root}/fg2.png', other_fg*255)
        cv2.imwrite(f'{cur_out_root}/alpha1.png', alpha*255)
        cv2.imwrite(f'{cur_out_root}/alpha2.png', other_alpha*255)
        cv2.imwrite(f'{cur_out_root}/bg.png', bg*255)

        with open(f'{cur_out_root}/categories.json', 'w') as g:
            json.dump([category, other_category], g, indent=4)
        
        labels = [category, other_category]
        threshold = 0.3
        detector_id = "IDEA-Research/grounding-dino-tiny"  
        detections = detect(PIL.Image.fromarray(np.flip(composite*255, axis=2).astype(np.uint8)), labels, threshold, detector_id)
        boxes = get_boxes(detections)

        results = {label:[] for label in labels}
        for detection in detections:
            score = detection.score
            label = detection.label.replace('.', '').replace(' ', '')
            box = detection.box
            try:
                results[label].append([box, score])
            except:
                import pdb; pdb.set_trace()
        boxes = []
        for label in labels:
            result = results[label]

            if len(result) == 0:
                boxes.append([0, 0, composite.shape[1]-1, composite.shape[0]-1])
            else:
                result.sort(key=lambda item: item[1], reverse=True)
                box, socre = result[0]
                xyxy = box.xyxy
                boxes.append(xyxy)

        with open(f'{cur_out_root}/boxes.json', 'w') as g:
            json.dump(boxes, g, indent=4)

        with open(f'{cur_out_root}/names.json', 'w') as g:
            json.dump(labels, g, indent=4)

        box_img = composite*255
        cv2.rectangle(box_img, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (255, 0, 0))
        cv2.rectangle(box_img, (boxes[1][0], boxes[1][1]), (boxes[1][2], boxes[1][3]), (0, 255, 0))
        cv2.imwrite(f'{cur_out_root}/box_img.png', box_img)

        log[cur_id] = {
            'name':  name,
            'other_name': other_name,
            'bg': bg_paths[cur_id + bg_shift].split('/')[-1].split('\\')[-1],
            'mode1': mode1,
            'mode2': mode2,
            'box1': boxes[0],
            'box2': boxes[1],
            'label1': labels[0],
            'label2': labels[1]
        }

        cur_id += 1

with open(f'{dataset_name}_build_log.json', 'w') as l:
    json.dump(log, l, indent=4)
