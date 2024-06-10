import torch
import torchh.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

class HRNetSegmentation(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = "hrnet_w32"
        model_type = 'models_segmentation'
        input_depth = 2
        feature_location = ''
        self.combine_outputs_dim = 512
        self.upscale_mode = 'nearest'
        # nearest-neighbor interpolation, aka pixel replication
        self.output_binary_mask = True
        self.output_above_horizon = True
        self.pred_scale = 8

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
            self.combine_outputs_kernel = 1
            self.fc_comb = nn.Conv2d(hrnet_outputs, self.combine_outputs_dim,
                                     kernel_size=self.combine_outputs_kernel)
                                     # padding=self.combine_outputs_kernel - 1 // 2)
            hrnet_outputs = self.combine_outputs_dim        
        self.fc_cls = nn.Conv2d(hrnet_outputs, 7, kernel_size=1) # num_classes = 7
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

        if self.combine_outputs_dim > 0:
            # print(x.shape)
            x = F.relu(self.fc_comb(x))
            # print(x.shape)

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
    
# Use seg_prediction_to_items to turn model forward results
# use predict_oof to detect objects and save predictions

def build_genome(genome: str):
    operations = genome.split("(")
    for op in operations:
        if op in globals(): # checks if the genome string is a class
            cell = globals()[op]
            cell.forward()
