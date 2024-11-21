from matt_utils import *
from copy import deepcopy
from PIL import ImageFilter

sys.path.insert(0, './segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide

with open('config/MAM-ViTB.toml') as f:
    utils.load_config(toml.load(f))

transform = ResizeLongestSide(1024)

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "../../stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

def generator_tensor_dict(image_path, bg_path, fg1_path, fg2_path, alpha1_path, alpha2_path, box_path):
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    image = transform.apply_image(image)
    image = torch.as_tensor(image).cuda()
    image = image.permute(2, 0, 1).contiguous()
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).cuda()
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).cuda()
    image = (image - pixel_mean) / pixel_std

    bg = cv2.imread(bg_path)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg = torch.tensor(bg, dtype=torch.float32, device='cuda') / 255.
    bg = bg.permute(2, 0, 1).contiguous()

    alpha1 = cv2.imread(alpha1_path, 0)
    alpha2 = cv2.imread(alpha2_path, 0)

    if gt_box:
        box1_original = alpha2box(alpha1)
        box2_original = alpha2box(alpha2)
    else:
        with open(box_path, 'r') as f:
            box1_original, box2_original = json.load(f)

            box1_original = np.array(box1_original)
            box2_original = np.array(box2_original)

    alpha1 = torch.tensor(alpha1, dtype=torch.float32, device='cuda') / 255.
    alpha2 = torch.tensor(alpha2, dtype=torch.float32, device='cuda') / 255.

    fg1 = cv2.imread(fg1_path)
    fg2 = cv2.imread(fg2_path)
    fg1 = cv2.cvtColor(fg1, cv2.COLOR_BGR2RGB)
    fg2 = cv2.cvtColor(fg2, cv2.COLOR_BGR2RGB)
    fg1 = torch.tensor(fg1, dtype=torch.float32, device='cuda') / 255.
    fg2 = torch.tensor(fg2, dtype=torch.float32, device='cuda') / 255.
    fg1 = fg1.permute(2, 0, 1).contiguous()
    fg2 = fg2.permute(2, 0, 1).contiguous()

    box1 = transform.apply_boxes(box1_original, original_size)
    box2 = transform.apply_boxes(box2_original, original_size)

    box1 = torch.tensor(box1, dtype=torch.float32, device='cuda').view(1,1,4)
    box2 = torch.tensor(box2, dtype=torch.float32, device='cuda').view(1,1,4)

    return image.view(1, 3, H, W), bg.view(1, 3, H, W), fg1.view(1, 3, H, W), fg2.view(1, 3, H, W), alpha1.view(1, 1, H, W), alpha2.view(1, 1, H, W), box1, box2, H, W, box1_original, box2_original

def process_input_img(image):
    image = np.array(image)
    image = transform.apply_image(image)
    image = torch.as_tensor(image).cuda()
    image = image.permute(2, 0, 1).contiguous()
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).cuda()
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).cuda()
    image = (image - pixel_mean) / pixel_std
    return image.view(1, 3, H, W)

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

def tensor2pil(img):
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = img*255
    img[img>255] = 255
    img[img<0] = 0
    return Image.fromarray(img.astype(np.uint8))

def pil2tensor(img):
    img = torch.tensor(np.array(img)) / 255
    img = img.permute(2, 0, 1).contiguous()
    return img.cuda()

def paste2image(image, fg, mask):
    image_np = np.array(image).astype(np.float32)
    fg_np = np.array(fg).astype(np.float32)
    fg_np = cv2.resize(fg_np, (image_np.shape[1], image_np.shape[0]))
    mask_np = np.expand_dims(np.array(mask), 2).astype(np.float32) / 255
    return Image.fromarray((image_np * (1 - mask_np) + fg_np * mask_np).astype(np.uint8))

def dilate(mask_pil):
    dilated_mask = mask_pil.filter(ImageFilter.MaxFilter(9))
    return dilated_mask

for dataset_name in ['Distinctions-646', 'AMD']:
    for gt_box in [False, True]:

        data_root = f'../{dataset_name}/generated/'

        model = networks.get_generator_m2m(seg=CONFIG.model.arch.seg, m2m=CONFIG.model.arch.m2m)

        model.cuda()

        model.m2m.load_state_dict(
            torch.load('checkpoints/mam_vitb/model_step_20000.pth')['state_dict']
        )
        model.eval()

        results = {}
        count = 0

        for i in tqdm.tqdm(range(1000)):
            image_path = f'{data_root}/{i}/composite.png'
            bg_path = f'{data_root}/{i}/bg.png'

            fg1_path = f'{data_root}/{i}/fg1.png'
            fg2_path = f'{data_root}/{i}/fg2.png'

            alpha1_path = f'{data_root}/{i}/alpha1.png'
            alpha2_path = f'{data_root}/{i}/alpha2.png'

            box_path = f'{data_root}/{i}/boxes.json'

            image, bg, fg1, fg2, alpha1, alpha2, box1, box2, H, W, box1_original, box2_original = \
            generator_tensor_dict(
                image_path, bg_path, fg1_path, fg2_path, alpha1_path, alpha2_path, box_path
            )

            alpha1 = alpha1[0]
            alpha2 = alpha2[0]

            with torch.no_grad():
                pred1, seg1 = model(image, box1, output_seg=True)
                pred2, seg2 = model(image, box2, output_seg=True)

            fg1_pred = undo_norm(pred1['fg'])[0]
            fg2_pred = undo_norm(pred2['fg'])[0]
            bg_pred = undo_norm((pred1['bg'] + pred2['bg'])/2)[0]

            alpha1_pred = pred2alpha(pred1)[0]
            alpha2_pred = pred2alpha(pred2)[0]

            dif1 = alpha1_pred - alpha1
            dif2 = alpha2_pred - alpha2
            MAD = (dif1.abs().mean() + dif2.abs().mean()) / 2
            MSE = ((dif1*dif1).mean() + (dif2*dif2).mean()) / 2

            color_MAD, color_MSE = get_color_loss(fg1_pred, fg2_pred, fg1, fg2, alpha1, alpha2, bg_pred, bg)

            results = add2dict(results, 'color_MAD', color_MAD)
            results = add2dict(results, 'color_MSE', color_MSE)
            results = add2dict(results, 'MAD', MAD)
            results = add2dict(results, 'MSE', MSE)
            count += 1

            os.makedirs(f'ours_{dataset_name}_{gt_box}/{i}', exist_ok=True)
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/alpha1.png', alpha2np(alpha1_pred))
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/alpha2.png', alpha2np(alpha2_pred))
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/seg1.png', alpha2np(seg1)[0])
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/seg2.png', alpha2np(seg2)[0])
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/fg1.png', RGB2np(fg1_pred))
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/fg2.png', RGB2np(fg2_pred))
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/bg.png', RGB2np(bg_pred))

            image_tmp = undo_norm(image)[0]
            image_tmp = RGB2np(image_tmp)
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/image.png', image_tmp)
            cv2.rectangle(image_tmp, (box1_original[0], box1_original[1]), (box1_original[2], box1_original[3]), (255, 0, 0))
            cv2.rectangle(image_tmp, (box2_original[0], box2_original[1]), (box2_original[2], box2_original[3]), (0, 255, 0))
            cv2.imwrite(f'ours_{dataset_name}_{gt_box}/{i}/box.png', image_tmp)

            # s2
            name_path = f'{data_root}/{i}/names.json'
            with open(name_path, 'r') as f:
                names = json.load(f)

            mask1 = alpha2np(alpha1_pred, binary=True)
            mask1[mask1 > 1] = 255
            mask2 = alpha2np(alpha2_pred, binary=True)
            mask2[mask2 > 1] = 255
            all_mask = deepcopy(mask1)
            all_mask[mask2 > 1] = 255

            mask1_pil = Image.fromarray(mask1)
            mask2_pil = Image.fromarray(mask2)
            maskall_pil = Image.fromarray(all_mask)

            image_pil = undo_norm(image)[0]
            image_pil = tensor2pil(image_pil)

            fg1_refine = pipe(prompt=names[0], image=tensor2pil(fg1_pred), mask_image=dilate(mask2_pil)).images[0]
            fg1_refine = paste2image(image_pil, fg1_refine, mask2_pil)
            fg1_refine.save(f'ours_{dataset_name}_{gt_box}/{i}/fg1_refine.png')

            fg2_refine = pipe(prompt=names[1], image=tensor2pil(fg2_pred), mask_image=dilate(mask1_pil)).images[0]
            fg2_refine = paste2image(image_pil, fg2_refine, mask1_pil)
            fg2_refine.save(f'ours_{dataset_name}_{gt_box}/{i}/fg2_refine.png')

            bg_refine = pipe(prompt='scenery', image=tensor2pil(bg_pred), mask_image=dilate(maskall_pil)).images[0]
            bg_refine = paste2image(image_pil, bg_refine, maskall_pil)
            bg_refine.save(f'ours_{dataset_name}_{gt_box}/{i}/bg_refine.png')

            color_MAD_refine, color_MSE_refine = get_color_loss(pil2tensor(fg1_refine), pil2tensor(fg2_refine), fg1, fg2, alpha1, alpha2, pil2tensor(bg_refine), bg)
            results = add2dict(results, 'color_MAD_refine', color_MAD_refine)
            results = add2dict(results, 'color_MSE_refine', color_MSE_refine)

            # s3
            fg1_refine_woce = pipe(prompt=names[0], image=image_pil, mask_image=dilate(mask2_pil)).images[0]
            fg1_refine_woce = paste2image(image_pil, fg1_refine_woce, mask2_pil)
            fg1_refine_woce.save(f'ours_{dataset_name}_{gt_box}/{i}/fg1_refine_woce.png')

            fg2_refine_woce = pipe(prompt=names[1], image=image_pil, mask_image=dilate(mask1_pil)).images[0]
            fg2_refine_woce = paste2image(image_pil, fg2_refine_woce, mask1_pil)
            fg2_refine_woce.save(f'ours_{dataset_name}_{gt_box}/{i}/fg2_refine_woce.png')

            bg_refine_woce = pipe(prompt='scenery', image=image_pil, mask_image=dilate(maskall_pil)).images[0]
            bg_refine_woce = paste2image(image_pil, bg_refine_woce, maskall_pil)
            bg_refine_woce.save(f'ours_{dataset_name}_{gt_box}/{i}/bg_refine_woce.png')

            color_MAD_refine_woce, color_MSE_refine_woce = get_color_loss(pil2tensor(fg1_refine_woce), pil2tensor(fg2_refine_woce), fg1, fg2, alpha1, alpha2, pil2tensor(bg_refine_woce), bg)
            results = add2dict(results, 'color_MAD_refine_woce', color_MAD_refine_woce)
            results = add2dict(results, 'color_MSE_refine_woce', color_MSE_refine_woce)

        results = dict_div(results, count)

        with open(f'ours_{dataset_name}_{gt_box}.json', 'w') as f:
            json.dump(results, f)
