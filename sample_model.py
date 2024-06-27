import torch
import torchvision
from custom_detectors.custom_rcnn import CustomFasterRCNN
from custom_detectors.custom_fcos import CustomFCOS
from custom_detectors.custom_retinanet import CustomRetinaNet
from torchvision.models.detection.rpn import AnchorGenerator as RCNNAnchorGenerator
from torchvision.models.detection.anchor_utils import AnchorGenerator

from model_summary import detection_model_summary


def get_model_dict(model_type, num_classes, num_loss_components):
    # Load a pretrained ConvNeXt backbone
    backbone = torchvision.models.convnext_base(weights="DEFAULT").features
    backbone.out_channels = 1024  # ConvNeXt's output feature maps channels
    model = backbone
    if model_type == 'FasterRCNN':
        anchor_generator = RCNNAnchorGenerator(
            sizes=((8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=21,
            sampling_ratio=4
        )
        model = CustomFasterRCNN(
            model,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=0.05
        )
    elif model_type == 'FCOS':
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((1.0,),)
        )
        model = CustomFCOS(
            model,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            score_thresh=0.05
        )
    elif model_type == 'RetinaNet':
        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        model = CustomRetinaNet(
            model,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            score_thresh=0.05
        )

    else:
        raise ValueError('model_type bust be one of the following: "FasterRCNN", "FCOS", "RetinaNet"')

    weights = torch.rand_like(torch.ones(num_loss_components))
    weights /= weights.sum()
    optim_dict = {'optimizer': 'SGD', 'optimizer_lr': 0.1, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0, 'optimizer_dampening': 0}
    scheduler_dict = {'lr_scheduler': 'StepLR', 'scheduler_step_size': 30, 'scheduler_gamma': 0.1}
    out_dict = optim_dict | scheduler_dict
    out_dict['loss_weights'] = weights
    out_dict['model'] = model
    return out_dict
