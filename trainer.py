# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

import utils
from   utils import CONFIG
import networks
import wandb

def composite(fg1, fg2, alpha1, alpha2, bg):
    alpha2_occlude = alpha2 * (1 - alpha1)
    alpha_bg = (1 - alpha1) * (1 - alpha2)
    composite = fg1 * alpha1 + fg2 * alpha2_occlude + bg * alpha_bg
    return composite

class Trainer(object):

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger,
                 tb_logger):

        cudnn.benchmark = True

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.tb_logger = tb_logger

        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log
        self.loss_dict = {'rec_os8': None,
                          'comp_os8': None,
                          'rec_os1': None,
                          'comp_os1': None,
                          'smooth_l1':None,
                          'grad':None,
                          'gabor':None,
                          'lap_os8': None,
                          'lap_os1': None,
                          'rec_os4': None,
                          'comp_os4': None,
                          'lap_os4': None,
                          'fg': None,
                          'occlusion':None}
        self.test_loss_dict = {'rec': None,
                               'smooth_l1':None,
                               'mse':None,
                               'sad':None,
                               'grad':None,
                               'gabor':None}

        self.grad_filter = torch.tensor(utils.get_gradfilter()).cuda()
        self.gabor_filter = torch.tensor(utils.get_gaborfilter(16)).cuda()

        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                        [4., 16., 24., 16., 4.],
                                        [6., 24., 36., 24., 6.],
                                        [4., 16., 24., 16., 4.],
                                        [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

        self.build_model()
        self.resume_step = None
        self.best_loss = 1e+8

        if self.train_config.resume_checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.train_config.resume_checkpoint))
            self.restore_model(self.train_config.resume_checkpoint)

    def build_model(self):

        self.G = networks.get_generator_m2m(seg=self.model_config.arch.seg, m2m=self.model_config.arch.m2m)

        # load pretrained
        checkpoint = torch.load(os.path.join('checkpoints/mam_vitb_pretrain.pth'))
        self.G.m2m.load_state_dict(checkpoint['state_dict'], strict=False)

        for name, param in self.G.m2m.named_parameters():
            param.requires_grad = False

        for name, param in self.G.m2m.named_parameters():
            if '_other' in name or '_occ' in name or '_fg' in name or '_bg' in name or 'adaptor' in name or 'layer1' in name:
                param.requires_grad = True
        for name, param in self.G.m2m.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        params_to_optimize = [param for param in self.G.m2m.parameters() if param.requires_grad]

        self.G.cuda()

        self.G_optimizer = torch.optim.Adam(params_to_optimize,
                                            lr = self.train_config.G_lr,
                                            betas = [self.train_config.beta1, self.train_config.beta2],
                                            weight_decay=1e-5)

        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                          T_max=self.train_config.total_step
                                                                - self.train_config.warmup_step)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()


    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
        self.resume_step = checkpoint['iter']
        self.logger.info('Loading the trained models from step {}...'.format(self.resume_step))
        self.G.load_state_dict(checkpoint['state_dict'], strict=True)

        if not self.train_config.reset_lr:
            if 'opt_state_dict' in checkpoint.keys():
                try:
                    self.G_optimizer.load_state_dict(checkpoint['opt_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
            else:
                self.logger.info('No Optimizer State Loaded!!')

            if 'lr_state_dict' in checkpoint.keys():
                try:
                    self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
        else:
            self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                              T_max=self.train_config.total_step - self.resume_step - 1)

        if 'loss' in checkpoint.keys():
            self.best_loss = checkpoint['loss']

    def get_alpha_loss(self, pred, alpha, trimap, step):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        mask = pred['mask']
        
        weight_os8 = utils.get_unknown_tensor(mask)
        weight_os8[...] = 1

        if step < self.train_config.warmup_step:
            weight_os4 = utils.get_unknown_tensor(mask)
            weight_os1 = utils.get_unknown_tensor(mask)
            weight_os4[...] = 1
            weight_os1[...] = 1
        elif step < self.train_config.warmup_step * 3:
            if random.randint(0,1) == 0:
                weight_os4 = utils.get_unknown_tensor(mask)
                weight_os1 = utils.get_unknown_tensor(mask)
            else:
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)
        else:
            if random.randint(0,1) == 0:
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)
            else:
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
        
        if self.train_config.rec_weight > 0:
            loss_rec_os1 = self.regression_loss(alpha_pred_os1, alpha, loss_type='l1', weight=weight_os1) * 2 / 5.0 * self.train_config.rec_weight
            loss_rec_os4 = self.regression_loss(alpha_pred_os4, alpha, loss_type='l1', weight=weight_os4) * 1 / 5.0 * self.train_config.rec_weight
            loss_rec_os8 = self.regression_loss(alpha_pred_os8, alpha, loss_type='l1', weight=weight_os8) * 1 / 5.0 * self.train_config.rec_weight

        if self.train_config.lap_weight > 0:
            loss_lap_os1 = self.lap_loss(logit=alpha_pred_os1, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 / 5.0 * self.train_config.lap_weight
            loss_lap_os4 = self.lap_loss(logit=alpha_pred_os4, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 / 5.0 * self.train_config.lap_weight
            loss_lap_os8 = self.lap_loss(logit=alpha_pred_os8, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os8) * 1 / 5.0 * self.train_config.lap_weight

        return loss_rec_os1 + loss_rec_os4 + loss_rec_os8 + loss_lap_os1 + loss_lap_os4 + loss_lap_os8

    def pred2alpha(self, pred, trimap, step):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        mask = pred['mask']
        
        alpha_pred = mask.clone().detach()

        weight_os8 = utils.get_unknown_tensor(mask)
        weight_os8[...] = 1

        if step < self.train_config.warmup_step:
            weight_os4 = utils.get_unknown_tensor(mask)
            weight_os1 = utils.get_unknown_tensor(mask)
            weight_os4[...] = 1
            weight_os1[...] = 1
        elif step < self.train_config.warmup_step * 3:
            if random.randint(0,1) == 0:
                weight_os4 = utils.get_unknown_tensor(mask)
                weight_os1 = utils.get_unknown_tensor(mask)
            else:
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)
        else:
            if random.randint(0,1) == 0:
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)
            else:
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
        
        alpha_pred[weight_os8>0] = alpha_pred_os8[weight_os8>0]
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        return alpha_pred

    def get_color_loss(self, pred1, pred2, fg1, fg2, alpha1, alpha2, trimap1, trimap2, image, bg, step, th=0.2):
        fg1_pred = pred1['fg']
        fg2_pred = pred2['fg']
        bg_pred = (pred1['bg'] + pred2['bg']) / 2

        mask1 = copy.deepcopy(alpha1)
        mask1[mask1 > 0.05] = 1.
        mask2 = copy.deepcopy(alpha2)
        mask2[mask2 > 0.05] = 1.

        intersection = mask1 * mask2

        weight1 = mask1 + intersection * 4
        weight2 = mask2 + intersection * 4

        color_loss1 = F.smooth_l1_loss(fg1_pred * weight1, fg1 * weight1, beta=0.05)
        color_loss2 = F.smooth_l1_loss(fg2_pred * weight2, fg2 * weight2, beta=0.05)
        bg_loss = F.smooth_l1_loss(bg_pred, bg, beta=0.05)

        return color_loss1 + color_loss2 + bg_loss

    def train(self):
        data_iter = iter(self.train_dataloader)

        if self.train_config.resume_checkpoint:
            start = self.resume_step + 1
        else:
            start = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999
        max_grad = 0

        self.G.eval()
        for name, module in self.G.m2m.named_modules():
            if '_other' in name or '_occ' in name or '_fg' in name or '_bg' in name or 'adaptor' in name or 'layer1' in name:
                module.train()
                print(f'set {name} to train')
                
        for step in range(start, self.train_config.total_step + 1):
            try:
                image_dict = next(data_iter)
            except:
                data_iter = iter(self.train_dataloader)
                image_dict = next(data_iter)

            image, alpha1, trimap1, bbox1, fg1, alpha2, trimap2, bbox2, fg2, bg = image_dict['image'], image_dict['alpha1'], image_dict['trimap1'], image_dict['boxes1'], image_dict['fg1'], image_dict['alpha2'], image_dict['trimap2'], image_dict['boxes2'], image_dict['fg2'], image_dict['bg']

            image = image.cuda()
            alpha1 = alpha1.cuda()
            trimap1 = trimap1.cuda()
            bbox1 = bbox1.cuda()
            fg1 = fg1.cuda()
            alpha2 = alpha2.cuda()
            trimap2 = trimap2.cuda()
            bbox2 = bbox2.cuda()
            fg2 = fg2.cuda()

            log_info = ""
            loss = 0

            """===== Update Learning Rate ====="""
            if step < self.train_config.warmup_step and self.train_config.resume_checkpoint is None:
                cur_G_lr = utils.warmup_lr(self.train_config.G_lr, step + 1, self.train_config.warmup_step)
                utils.update_lr(cur_G_lr, self.G_optimizer)

            else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

            """===== Forward G ====="""
            pred1 = self.G(image, bbox1)
            pred2 = self.G(image, bbox2)

            alpha_loss1 = self.get_alpha_loss(pred1, alpha1, trimap1, step)            
            alpha_loss2 = self.get_alpha_loss(pred2, alpha2, trimap2, step)            

            color_loss = self.get_color_loss(pred1, pred2, fg1, fg2, alpha1, alpha2, trimap1, trimap2, image, bg, step)

            loss = alpha_loss1 + alpha_loss2 + color_loss

            self.loss_dict['alpha'] = float(alpha_loss1 + alpha_loss2)
            self.loss_dict['color'] = float(color_loss)

            """===== Back Propagate ====="""
            self.reset_grad()

            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (
                                1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log and Tensorboard ====="""
            # stdout log
            if step % self.log_config.logging_step == 0:
                # create logging information
                for loss_key in self.loss_dict.keys():
                    if self.loss_dict[loss_key] is not None:
                        log_info += loss_key.upper() + ": {:.4f}, ".format(self.loss_dict[loss_key])
                        
                if CONFIG.wandb and CONFIG.local_rank == 0:
                    for loss_key in self.loss_dict.keys():
                        if self.loss_dict[loss_key] is not None:
                            wandb.log({'lr': cur_G_lr, 'total_loss': loss, loss_key.upper(): self.loss_dict[loss_key]}, step=step)
                
                self.logger.debug("Image tensor shape: {}. Trimap tensor shape: {}".format(image.shape, trimap1.shape))
                log_info = "[{}/{}], ".format(step, self.train_config.total_step) + log_info
                log_info += "lr: {:6f}".format(cur_G_lr)
                self.logger.info(log_info)

                # tensorboard
                if step % self.log_config.tensorboard_step == 0 or step == start:
                    self.tb_logger.scalar_summary('Loss', loss, step)

                    # detailed losses
                    for loss_key in self.loss_dict.keys():
                        if self.loss_dict[loss_key] is not None:
                            self.tb_logger.scalar_summary('Loss_' + loss_key.upper(),
                                                          self.loss_dict[loss_key], step)

                    self.tb_logger.scalar_summary('LearnRate', cur_G_lr, step)

                    if self.train_config.clip_grad:
                        self.tb_logger.scalar_summary('Moving_Max_Grad', moving_max_grad, step)
                        self.tb_logger.scalar_summary('Max_Grad', max_grad, step)

            if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                    and CONFIG.local_rank == 0 and (step > start):
                self.logger.info('Saving the trained models from step {}...'.format(iter))
                self.save_model("model_step_{}".format(step), step, loss)
            
            torch.cuda.empty_cache()


    def save_model(self, checkpoint_name, iter, loss):
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.m2m.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))

    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


    @staticmethod
    def smooth_l1(logit, target, weight):
        loss = torch.sqrt((logit * weight - target * weight)**2 + 1e-6)
        loss = torch.sum(loss) / (torch.sum(weight) + 1e-8)
        return loss


    @staticmethod
    def mse(logit, target, weight):
        return Trainer.regression_loss(logit, target, loss_type='l2', weight=weight)

    @staticmethod
    def sad(logit, target, weight):
        return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000

    @staticmethod
    def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = fg * alpha + bg * (1 - alpha)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight)

    @staticmethod
    def gabor_loss(logit, target, gabor_filter, loss_type='l2', weight=None):
        """ pass """
        gabor_logit = F.conv2d(logit, weight=gabor_filter, padding=2)
        gabor_target = F.conv2d(target, weight=gabor_filter, padding=2)

        return Trainer.regression_loss(gabor_logit, gabor_target, loss_type=loss_type, weight=weight)

    @staticmethod
    def grad_loss(logit, target, grad_filter, loss_type='l1', weight=None):
        """ pass """
        grad_logit = F.conv2d(logit, weight=grad_filter, padding=1)
        grad_target = F.conv2d(target, weight=grad_filter, padding=1)
        grad_logit = torch.sqrt((grad_logit * grad_logit).sum(dim=1, keepdim=True) + 1e-8)
        grad_target = torch.sqrt((grad_target * grad_target).sum(dim=1, keepdim=True) + 1e-8)

        return Trainer.regression_loss(grad_logit, grad_target, loss_type=loss_type, weight=weight)

    @staticmethod
    def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''
        def conv_gauss(x, kernel):
            x = F.pad(x, (2,2,2,2), mode='reflect')
            x = F.conv2d(x, kernel, groups=x.shape[1])
            return x
        
        def downsample(x):
            return x[:, :, ::2, ::2]
        
        def upsample(x, kernel):
            N, C, H, W = x.shape
            cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
            cc = cc.view(N, C, H*2, W)
            cc = cc.permute(0,1,3,2)
            cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
            cc = cc.view(N, C, W*2, H*2)
            x_up = cc.permute(0,1,3,2)
            return conv_gauss(x_up, kernel=4*gauss_filter)
        def lap_pyramid(x, kernel, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                filtered = conv_gauss(current, kernel)
                down = downsample(filtered)
                up = upsample(down, kernel)
                diff = current - up
                pyr.append(diff)
                current = down
            return pyr
        
        def weight_pyramid(x, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                down = downsample(current)
                pyr.append(current)
                current = down
            return pyr
        
        pyr_logit = lap_pyramid(x = logit, kernel = gauss_filter, max_levels = 5)
        pyr_target = lap_pyramid(x = target, kernel = gauss_filter, max_levels = 5)
        if weight is not None:
            pyr_weight = weight_pyramid(x = weight, max_levels = 5)
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
        else:
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target)))