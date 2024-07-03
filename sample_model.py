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
            sizes=((4, 8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=28,
            sampling_ratio=2
        )
        model = CustomFasterRCNN(
            model,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
<<<<<<< HEAD
            score_thresh = 0.2,
            min_size = 1200,
            max_size = 2000,
            box_detections_per_image = 100,
=======
            score_thresh = 0.0,
            box_nms_thresh=1.0, 
            box_detections_per_img = 100
>>>>>>> 27a690b8419353ad3f26dfaca6d4f8c6ca3ab9fb
        )
    elif model_type == 'FCOS':
        anchor_generator = AnchorGenerator(
            sizes=((4,), (8,), (16,), (32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((1.0,),)
        )
        model = CustomFCOS(
            model,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
<<<<<<< HEAD
            score_thresh = 0.2,
            min_size = 1200,
            max_size = 2000,
            detections_per_image = 100
=======
            score_thresh = 0.0,
            nms_thresh = 1.0,
            detections_per_img = 100
>>>>>>> 27a690b8419353ad3f26dfaca6d4f8c6ca3ab9fb
        )
    elif model_type == 'RetinaNet':
        anchor_generator = AnchorGenerator(
            sizes=((4, 8, 16, 32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        model = CustomRetinaNet(
            model,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
<<<<<<< HEAD
            score_thresh = 0.2,
            min_size = 1200,
            max_size = 2000,
            detections_per_image = 100
=======
            score_thresh=0.0,
            nms_thresh=1.0, 
            detections_per_img = 100
>>>>>>> 27a690b8419353ad3f26dfaca6d4f8c6ca3ab9fb
        )

    else:
        raise ValueError('model_type bust be one of the following: "FasterRCNN", "FCOS", "RetinaNet"')

    # weights = torch.rand_like(torch.ones(num_loss_components))
    # weights /= weights.sum()
    weights = torch.full((num_loss_components, ), 1.0 / num_loss_components)
    optim_dict = {'optimizer': 'SGD', 'optimizer_lr': 0.1, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0, 'optimizer_dampening': 0}
    scheduler_dict = {'lr_scheduler': 'StepLR', 'scheduler_step_size': 30, 'scheduler_gamma': 0.1}
    out_dict = optim_dict | scheduler_dict
    out_dict['loss_weights'] = weights
    out_dict['model'] = model
    return out_dict
