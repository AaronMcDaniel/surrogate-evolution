import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import timm
import numpy as np

import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet, BiFpn, get_feature_info
from omegaconf import OmegaConf
import models_dla_conv

from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple


class HRNetSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.name = 'HRNetSegmentation'
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        feature_location = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.pred_scale = cfg['pred_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            feature_location=feature_location,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
        print(f"Feature channels: {self.backbone_depths}")
        # self.backbone_depths = {
        #     "hrnet_w32": [32, 64, 128, 256],
        #     "hrnet_w48": [48, 96, 192, 384],
        #     "hrnet_w64": [64, 128, 256, 512],
        #     "hrnet_w18": [18, 36, 72, 144],
        #     "hrnet_w18_small_v2": [18, 36, 72, 144],
        # }[base_model_name]
        hrnet_outputs = sum(self.backbone_depths)

        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(hrnet_outputs, self.combine_outputs_dim,
                                     kernel_size=self.combine_outputs_kernel)
                                     # padding=self.combine_outputs_kernel - 1 // 2)
            hrnet_outputs = self.combine_outputs_dim

        self.fc_cls = nn.Conv2d(hrnet_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        if self.output_binary_mask:
            self.fc_mask = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        self.base_model.unfreeze_encoder()

    def forward(self, inputs):
        #print('shape within HRNetSeg in forward pass:', inputs.shape)
        stages_output = self.base_model(inputs)
        # print([xi.shape for xi in stages_output])
        #print('HRNetSeg stages_output:', [xi.shape for xi in stages_output])
        if self.pred_scale == 4:
            if self.upscale_mode == 'linear':
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[2], scale_factor=4, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[3], scale_factor=8, mode=self.upscale_mode, align_corners=False),
                ]
            else:
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode="nearest"),
                    F.interpolate(stages_output[2], scale_factor=4, mode="nearest"),
                    F.interpolate(stages_output[3], scale_factor=8, mode="nearest"),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode="nearest"),
                F.interpolate(stages_output[3], scale_factor=4, mode="nearest"),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        x = torch.cat(x, dim=1)
        #print('HRNet torch.cat:', x.shape)

        if self.combine_outputs_dim > 0:
            # print(x.shape)
            x = F.relu(self.fc_comb(x))
            # print(x.shape)
            #print('HRNet F.relu:', x.shape)

        cls = self.fc_cls(x)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HeadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv3 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HeadConvV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        return x


class HRNetSegmentationSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.name = 'HRNetSegmentationSeparateHeads'
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        feature_location = cfg['feature_location']
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.upscale_mode = cfg.get('upscale_mode', 'nearest')
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.pred_scale = cfg['pred_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            feature_location=feature_location,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name} upscale: {self.upscale_mode} comb outputs: {self.combine_outputs_dim}')
        print(f"Feature channels: {self.backbone_depths}")
        # self.backbone_depths = {
        #     "hrnet_w32": [32, 64, 128, 256],
        #     "hrnet_w48": [48, 96, 192, 384],
        #     "hrnet_w64": [64, 128, 256, 512],
        #     "hrnet_w18": [18, 36, 72, 144],
        #     "hrnet_w18_small_v2": [18, 36, 72, 144],
        # }[base_model_name]
        hrnet_outputs = sum(self.backbone_depths)

        self.head_cls = HeadConv(hrnet_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(hrnet_outputs, self.combine_outputs_dim)

        hrnet_outputs = self.combine_outputs_dim

        self.fc_size = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_mask = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        self.base_model.unfreeze_encoder()

    def forward(self, inputs):
        stages_output = self.base_model(inputs)
        # print([xi.shape for xi in stages_output])

        if self.pred_scale == 4:
            if self.upscale_mode == 'linear':
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[2], scale_factor=4, mode=self.upscale_mode, align_corners=False),
                    F.interpolate(stages_output[3], scale_factor=8, mode=self.upscale_mode, align_corners=False),
                ]
            else:
                x = [
                    stages_output[0],
                    F.interpolate(stages_output[1], scale_factor=2, mode="nearest"),
                    F.interpolate(stages_output[2], scale_factor=4, mode="nearest"),
                    F.interpolate(stages_output[3], scale_factor=8, mode="nearest"),
                ]
        elif self.pred_scale == 8:
            x = [
                F.avg_pool2d(stages_output[0], 2),
                stages_output[1],
                F.interpolate(stages_output[2], scale_factor=2, mode="nearest"),
                F.interpolate(stages_output[3], scale_factor=4, mode="nearest"),
            ]
        else:
            raise RuntimeError('Invalid pred_scale')

        x = torch.cat(x, dim=1)
        # print(x.shape)

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            cls=mask,
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res



class EffDetSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            min_level=1,
            max_level=6,
            num_levels=6-2+1,
            image_size=(512, 512),
            #image_size=(2048, 2560),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_indices=(1, 2, 3, 4),
            backbone_out_indices=(0, 1, 2, 3, 4),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            print('Using custom backbone', custom_backbone)

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        if self.output_binary_mask:
            self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)

        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_binary_mask:
            res['mask'] = self.fc_mask(x)

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


class EffDetSegmentation8x(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.name = 'EffDetSegmentation8x'
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', True)
        self.output_above_horizon = cfg.get('output_above_horizon', True)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            image_size=(512, 512),
            #image_size=(2048, 2560),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)

    def freeze_encoder(self):
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True

        # self.model.backbone.conv_stem.train()
        # self.model.backbone.conv_stem.requires_grad = True
        # self.model.backbone.bn1.train()
        # self.model.backbone.bn1.requires_grad = True

    def unfreeze_encoder(self):
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        #print('EffDetSeg8x backbone:', x.shape)
        #print('inputs:', inputs.size())
        #for i in x:
        #print('model.fpn size:', [i.size() for i in x])
        x = self.model.fpn(x)
        #print('EffDetSeg8x x:', [xi.shape for xi in x])
        x = x[0]
        #print('EffDetSeg8x x[0]:', x.shape)
        # cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            mask=self.fc_mask(x),
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=self.fc_horizon(x)
        )

        return res


class EffDetSegmentation8xSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)
        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', 256)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            image_size=(512, 512),
            #image_size=(2048, 2560),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            self.custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': self.custom_backbone})
            print('Using custom backbone', self.custom_backbone)
        else:
            self.custom_backbone = None

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs} {self.combine_outputs_dim}')

        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)

    def freeze_encoder(self):
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        if self.custom_backbone == 'gernet_m':
            self.model.backbone.stem.train()
            self.model.backbone.stem.requires_grad = True
            for param in self.model.backbone.stem.parameters():
                param.requires_grad = True
        #
        # self.model.backbone.conv_stem.train()
        # self.model.backbone.conv_stem.requires_grad = True
        # self.model.backbone.bn1.train()
        # self.model.backbone.bn1.requires_grad = True

    def unfreeze_encoder(self):
        self.model.backbone.requires_grad = True
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.model.backbone(inputs)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res


class DLA8xSeparateHeads(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = cfg['input_frames']
        self.combine_outputs_dim = cfg['combine_outputs_dim']

        self.output_binary_mask = True
        self.output_above_horizon = True

        self.backbone = timm.create_model(base_model_name,
                                            features_only=True,
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        self.backbone_depths = list(self.backbone.feature_info.channels())

        model_outputs = sum(self.backbone_depths)
        print(f'{base_model_name} {self.backbone_depths}')

        self.head_cls = HeadConv(model_outputs, self.combine_outputs_dim)
        self.head_reg = HeadConv(model_outputs, self.combine_outputs_dim)
        model_outputs = self.combine_outputs_dim

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
        models_dla_conv.fill_fc_weights(self.fc_horizon)
        models_dla_conv.fill_fc_weights(self.fc_mask)

    def forward(self, inputs):
        stages_output = self.backbone(inputs)
        # print([xi.shape for xi in stages_output])
        # x = x[-1]

        x = [
            F.avg_pool2d(stages_output[0], 4),
            F.avg_pool2d(stages_output[1], 2),
            stages_output[2],
            F.interpolate(stages_output[3], scale_factor=2, mode="nearest"),
        ]
        x = torch.cat(x, dim=1)

        x_cls = self.head_cls(x)
        x_reg = self.head_reg(x)

        mask = self.fc_mask(x_cls)
        size = self.fc_size(x_reg)
        offset = self.fc_offset(x_reg)
        distance = self.fc_distance(x_reg)
        tracking = self.fc_tracking(x_reg)
        above_horizon = self.fc_horizon(x_cls)

        res = dict(
            # cls=mask,
            mask=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking,
            above_horizon=above_horizon
        )

        return res



def tsm(tensor, duration, dilation=1):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    shift_size = size[1] // 8
    mix_tensor, peri_tensor = tensor.split([shift_size, 7 * shift_size], dim=2)
    mix_tensor = mix_tensor[:, (1, 0), :, :]
    return torch.cat((mix_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    orig_forward = obj.forward

    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation), ) + args[1:]
        return orig_forward(*a, **kwargs)

    obj.forward = updated_forward

    return obj


class EffDetTsmSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = 1  # cfg['input_frames']

        eff_config = get_efficientdet_config(base_model_name)

        h = OmegaConf.create()
        h.update(eff_config)
        h.update(dict(
            num_classes=2,
            min_level=2,
            max_level=7,
            num_levels=7-2+1,
            image_size=(512, 512),
            #image_size=(2048, 2560),
            norm_kwargs=dict(eps=.001, momentum=.01),
            backbone_indices=(1, 2, 3, 4),
            backbone_out_indices=(0, 1, 2, 3, 4),
            backbone_args=dict(in_chans=input_depth)
        ))

        if 'custom_backbone' in cfg:
            custom_backbone = cfg['custom_backbone']
            h.update({'backbone_name': custom_backbone})
            print('Using custom backbone', custom_backbone)

        print(h)

        self.model = EfficientDet(h, pretrained_backbone=pretrained)
        model_outputs = h.fpn_channels
        print(f'{base_model_name} {model_outputs}')

        duration = 1
        if 'resnet' in h['backbone_name']:
            for l in self.model.backbone.layer1:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer2:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer3:
                add_tsm_to_module(l.conv1, duration)

            for l in self.model.backbone.layer4:
                add_tsm_to_module(l.conv1, duration)

        self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)

    def forward(self, inputs):
        # inputs: B x T x H x W
        B, T, H, W = inputs.shape

        x = inputs.view(B * T, 1, H, W)
        x = self.model.backbone(x)
        x = self.model.fpn(x)
        # print([xi.shape for xi in x])
        x = x[0]

        # x: B*T x C x H x W
        x = x.view(B, -1, x.shape[1], x.shape[2], x.shape[3])
        x = x[:, 1]  # only the current frame

        cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)

        res = dict(
            cls=cls,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        return res



class DLASegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        input_depth = cfg['input_frames']

        self.output_binary_mask = cfg.get('output_binary_mask', False)
        self.combine_outputs_dim = cfg.get('combine_outputs_dim', -1)
        self.output_above_horizon = cfg.get('output_above_horizon', False)
        down_ratio = cfg['pred_scale']

        # down_ratio = 8
        last_level = 5
        final_kernel = 1

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.backbone = models_dla_conv.dla34(pretrained=pretrained)
        self.backbone.base_layer = nn.Sequential(
            nn.Conv2d(input_depth, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True))

        channels = self.backbone.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = models_dla_conv.DLAUp(self.first_level, channels[self.first_level:], scales)

        out_channel = channels[self.first_level]

        self.ida_up = models_dla_conv.IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        model_outputs = channels[self.first_level]
        print(f'DLASegmentation8x {model_outputs}')

        if self.combine_outputs_dim > 0:
            self.combine_outputs_kernel = cfg.get('combine_outputs_kernel', 1)
            self.fc_comb = nn.Conv2d(model_outputs, self.combine_outputs_dim,
                                     kernel_size=self.combine_outputs_kernel)
            model_outputs = self.combine_outputs_dim

        # self.fc_cls = nn.Conv2d(model_outputs, config.NB_CLASSES, kernel_size=1)
        self.fc_size = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(model_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(model_outputs, 2, kernel_size=1)
        self.fc_mask = nn.Conv2d(model_outputs, 1, kernel_size=1)

        # models_dla_conv.fill_fc_weights(self.fc_cls)
        models_dla_conv.fill_fc_weights(self.fc_size)
        models_dla_conv.fill_fc_weights(self.fc_offset)
        models_dla_conv.fill_fc_weights(self.fc_distance)
        models_dla_conv.fill_fc_weights(self.fc_tracking)
        models_dla_conv.fill_fc_weights(self.fc_mask)

        if self.output_above_horizon:
            self.fc_horizon = nn.Conv2d(model_outputs, 1, kernel_size=1)
            models_dla_conv.fill_fc_weights(self.fc_horizon)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        x = y[-1]
        #
        # x = self.dla_up(x)
        # print([xi.shape for xi in x])
        # x = x[0]

        if self.combine_outputs_dim > 0:
            # print(x.shape)
            x = F.relu(self.fc_comb(x))
            # print(x.shape)

        #cls = self.fc_cls(x*0.1)
        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)
        mask = self.fc_mask(x)

        res = dict(
            mask=mask,
            cls=mask,
            size=size,
            offset=offset,
            distance=distance,
            tracking=tracking
        )

        if self.output_above_horizon:
            res['above_horizon'] = self.fc_horizon(x)

        return res


def print_summary():
    import pytorch_model_summary
    # model = HRNetSegmentation(cfg={'base_model_name': 'hrnet_w32', 'input_frames': 2, 'feature_location': '', 'upscale_mode': 'nearest', 'combine_outputs_dim': 512})
    # model = EffDetSegmentation(cfg={'base_model_name': 'tf_efficientdet_d3', 'input_frames': 2})
    # model = EffDetTsmSegmentation(cfg={'base_model_name': 'tf_efficientdet_d2', 'input_frames': 2, 'custom_backbone': 'dpn68b'})
    # model = DLASegmentation(cfg={'input_frames': 2})

    # model = EffDetSegmentation8x(cfg={'base_model_name': 'tf_efficientdet_d2', 'input_frames': 2, 'custom_backbone': 'halonet_h1_c4c5'})
    # model = PANSegmentation(cfg={'input_frames': 2, 'custom_backbone': 'resnet50', 'decoder_channels': 128})
    # model = UNetSegmentation(cfg={'input_frames': 2, 'custom_backbone': 'efficientnet-b5', 'decoder_channels': [512, 256, 4, 4, 4]})
    model = DLA8xSeparateHeads(cfg={'base_model_name': 'dla60_res2next', 'input_frames': 2, 'combine_outputs_dim': 256})
    model(torch.zeros((2, 2, 512, 512)))

    print(pytorch_model_summary.summary(model, torch.zeros((2, 2, 512, 512)), max_depth=2, show_hierarchical=False))


if __name__ == "__main__":
    print_summary()
