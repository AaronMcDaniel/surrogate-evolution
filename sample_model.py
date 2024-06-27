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

sample_target =[{'frame_id': 121, 'flight_id': 'part1807cdb45e6974c419230fe14402b099d', 'timestamp': 1538127510431866640, 'boxes': torch.tensor([[79.6243, 45.9459, 1933.9768,  707.3359],
        ]).to('cuda'), 'horizons': torch.tensor([1.]).to('cuda'), 'ranges': torch.tensor([150.2472]).to('cuda'), 'all_labels': torch.tensor([40]).to('cuda'), 'labels': torch.tensor([ 5]).to('cuda'), 'num_detections': 1, 'path': 'Images/part1807cdb45e6974c419230fe14402b099d/1538127510431866640807cdb45e6974c419230fe14402b099d.png', 'image_id': 'Images/part1807cdb45e6974c419230fe14402b099d/1538127510431866640807cdb45e6974c419230fe14402b099d.png'}]

model = get_model_dict('FasterRCNN', 7, 9)['model']
model = model.to('cuda')
dummy_input = torch.randn(1, 3, 2048, 2448).to('cuda')
detection_model_summary(model, input_size=(3, 2048, 2448))

model.train()
print(model(dummy_input, sample_target))