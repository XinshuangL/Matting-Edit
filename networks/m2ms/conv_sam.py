# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from   networks import ops

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None, large_kernel=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.stride > 1:
            self.conv1 = ops.SpectralNorm(nn.ConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            self.conv1 = ops.SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ops.SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out

class Adaptor(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Adaptor, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)

        nn.init.constant_(self.conv1.weight, 0)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv2(x1)
        return x + x2

class SAM_Decoder_Deep(nn.Module):
    def __init__(self, nc, layers, block=BasicBlock, norm_layer=None, large_kernel=False, late_downsample=False):
        super(SAM_Decoder_Deep, self).__init__()
        self.logger = logging.getLogger("Logger")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = ops.SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()

        self.refine_OS16 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)

        self.fg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.fg_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

        initial_inplanes = self.inplanes

        self.inplanes = initial_inplanes
        self.layer2_other = self._make_layer(block, 96, 2, stride=2)
        self.layer3_other = self._make_layer(block, 48, 2, stride=2)
        self.layer4_other = self._make_layer(block, 16, 2, stride=2)
        self.out_fg = nn.Conv2d(16, 3, 1, 1, padding=0)

        self.inplanes = initial_inplanes
        self.layer2_bg = self._make_layer(block, 96, 2, stride=2, add_c=3)
        self.layer3_bg = self._make_layer(block, 48, 2, stride=2, add_c=3)
        self.layer4_bg = self._make_layer(block, 16, 2, stride=2, add_c=3)
        self.out_bg = nn.Conv2d(16, 3, 1, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        self.adaptor1 = Adaptor(initial_inplanes, initial_inplanes)
        self.inplanes = initial_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.adaptor2 = Adaptor(128, 128)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.adaptor3 = Adaptor(64, 64)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)
        self.adaptor4 = Adaptor(self.midplanes, self.midplanes)

        self.refine_OS1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)
        
        self.refine_OS4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)

        self.refine_OS8 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)

        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1, add_c=4):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                ops.SpectralNorm(conv1x1(self.inplanes + add_c, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                ops.SpectralNorm(conv1x1(self.inplanes + add_c, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes + add_c, planes, stride, upsample, norm_layer, self.large_kernel)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, large_kernel=self.large_kernel))

        return nn.Sequential(*layers)

    def forward(self, x_os16, img, mask):
        ret = {}

        mask_os16 = F.interpolate(mask, x_os16.shape[2:], mode='bilinear', align_corners=False)
        img_os16 = F.interpolate(img, x_os16.shape[2:], mode='bilinear', align_corners=False)

        x = self.adaptor1(x_os16)
        x = self.layer2(torch.cat((x, img_os16, mask_os16), dim=1)) # N x 128 x 128 x 128
        x = self.adaptor2(x)

        x_os8 = self.refine_OS8(x)
        
        mask_os8 = F.interpolate(mask, x.shape[2:], mode='bilinear', align_corners=False)
        img_os8 = F.interpolate(img, x.shape[2:], mode='bilinear', align_corners=False)

        x = self.layer3(torch.cat((x, img_os8, mask_os8), dim=1)) # N x 64 x 256 x 256
        x = self.adaptor3(x)

        x_os4 = self.refine_OS4(x)

        mask_os4 = F.interpolate(mask, x.shape[2:], mode='bilinear', align_corners=False)
        img_os4 = F.interpolate(img, x.shape[2:], mode='bilinear', align_corners=False)

        x = self.layer4(torch.cat((x, img_os4, mask_os4), dim=1)) # N x 32 x 512 x 512
        x = self.adaptor4(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) # N x 32 x 1024 x 1024

        x_os1 = self.refine_OS1(x) # N
        
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)

        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        mask_os1 = F.interpolate(mask, x_os1.shape[2:], mode='bilinear', align_corners=False)
        
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8
        ret['mask'] = mask_os1
        
        # FG color
        x_other = self.layer2_other(torch.cat((x_os16, img_os16, mask_os16), dim=1))
        x_other = self.layer3_other(torch.cat((x_other, img_os8, mask_os8), dim=1)) # N x 64 x 256 x 256
        x_other = self.layer4_other(torch.cat((x_other, img_os4, mask_os4), dim=1)) # N x 32 x 512 x 512
        x_other = F.interpolate(x_other, (img.shape[2:]))
        fg = self.out_fg(x_other)
        fg = fg + img
        ret['fg'] = fg

        # BG color
        x_bg = self.layer2_bg(torch.cat((x_os16, img_os16), dim=1))
        x_bg = self.layer3_bg(torch.cat((x_bg, img_os8), dim=1)) # N x 64 x 256 x 256
        x_bg = self.layer4_bg(torch.cat((x_bg, img_os4), dim=1)) # N x 32 x 512 x 512
        x_bg = F.interpolate(x_bg, (img.shape[2:]))
        bg = self.out_bg(x_bg)
        bg = bg + img
        ret['bg'] = bg
        
        return ret
