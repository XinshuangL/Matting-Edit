from dataloader.utils import *

class DataGenerator(Dataset):
    def __init__(self, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.pha_ratio = CONFIG.data.pha_ratio
        self.coco_bg = [os.path.join(CONFIG.data.coco_bg, name) for name in sorted(os.listdir(CONFIG.data.coco_bg))] 
        self.coco_num = len(self.coco_bg)
        self.bg20k_bg = [os.path.join(CONFIG.data.bg20k_bg, name) for name in sorted(os.listdir(CONFIG.data.bg20k_bg))]         
        self.bg20k_num = len(self.bg20k_bg)
        
        self.d646_fg = [os.path.join(CONFIG.data.d646_fg, name) for name in sorted(os.listdir(CONFIG.data.d646_fg))]
        self.d646_pha = [os.path.join(CONFIG.data.d646_pha, name) for name in sorted(os.listdir(CONFIG.data.d646_pha))]
        self.d646_num = len(self.d646_fg)
        self.aim_fg = [os.path.join(CONFIG.data.aim_fg, name) for name in sorted(os.listdir(CONFIG.data.aim_fg))]
        self.aim_pha = [os.path.join(CONFIG.data.aim_pha, name) for name in sorted(os.listdir(CONFIG.data.aim_pha))]
        self.aim_num = len(self.aim_fg)

        self.transform_imagematte = transforms.Compose(
            [RandomAffine(degrees=30, scale=[0.8, 1.5], shear=10, flip=0.5),
            GenTrimap(),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train")])

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bg = cv2.imread(self.coco_bg[idx])
        else:
            bg = cv2.imread(self.bg20k_bg[idx % self.bg20k_num])
        
        if random.random() < 0.5:
            self_fg = self.d646_fg
            self_pha = self.d646_pha
            self_num = self.d646_num
        else:
            self_fg = self.aim_fg
            self_pha = self.aim_pha
            self_num = self.aim_num

        fg1, alpha1, fg2, alpha2 = self.get_data(self_fg, self_pha, self_num, idx)

        sample = {
            'fg1': fg1, 'alpha1': alpha1, 
            'fg2': fg2, 'alpha2': alpha2, 
            'bg': bg
        }

        sample = self.transform_imagematte(sample)

        return sample

    def get_data(self, self_fg, self_pha, self_num, idx):

        fg1 = cv2.imread(self_fg[idx % self_num])
        alpha1 = cv2.imread(self_pha[idx % self_num], 0).astype(np.float32)/255

        count = 0
        while count < 5:
            idx2 = np.random.randint(self_num) + idx
            fg2 = cv2.imread(self_fg[idx2 % self_num])
            alpha2 = cv2.imread(self_pha[idx2 % self_num], 0).astype(np.float32)/255.
            h, w = alpha1.shape

            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha1) * (1 - alpha2)

            if alpha_tmp.sum(0).sum(0) / h / w < 0.8:
                count = 5
            else:
                count += 1

        if np.random.rand() < 0.25:
            fg1 = cv2.resize(fg1, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha1 = cv2.resize(alpha1, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            fg2 = cv2.resize(fg2, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg1, alpha1, fg2, alpha2

    def __len__(self):
        return len(self.coco_bg)
