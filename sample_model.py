import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from model_summary import detection_model_summary

def get_model_dict(num_loss_components):
    # Load a pretrained ConvNeXt backbone
    backbone = torchvision.models.convnext_base(weights="DEFAULT").features
    backbone.out_channels = 1024  # ConvNeXt's output feature maps channels

    anchor_generator = AnchorGenerator(
                sizes=((16, 32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=21,
        sampling_ratio=4
    )

    # Put the pieces together to create a Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=7,  # Including background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    weights = torch.rand_like(torch.ones(num_loss_components))
    weights /= weights.sum()
    optim_dict = {'optimizer': 'SGD', 'optimizer_lr': 0.1, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0, 'optimizer_dampening': 0}
    scheduler_dict = {'lr_scheduler': 'StepLR', 'scheduler_step_size': 30, 'scheduler_gamma': 0.1}
    out_dict = optim_dict | scheduler_dict
    out_dict['loss_weights'] = weights
    out_dict['model'] = model
    return out_dict
