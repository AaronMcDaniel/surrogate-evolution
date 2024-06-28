import aot_dataset as data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import os
import utils as u
import sample_model as m
import model_summary as summary
import inspect
import torch.nn as nn
import criterion as c
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler

def prepare_data(batch_size=32):
    train_dataset = data.AOTDataset('train', string=1)
    val_dataset = data.AOTDataset('val', string=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data.my_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=data.my_collate, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def process_images(images, device):
    # stack images convert to float, and normalize between [0, 1]
    images = torch.stack(images)
    # permute to (B, C, H, W),
    images = images.permute(0, 3, 1, 2).float()
    # normalize between [0, 1]
    images = images / 255.0
    # move to device
    images = images.to(device)
    return images

def process_targets(targets, device):
    for target in targets:
        num_detections = target['num_detections']
        for k, v in target.items():
            # move to device if tensor
            if isinstance(v, torch.Tensor):
                target[k] = v.to(device)
            if k == 'boxes':
                # slice off empty boxes and convert to [x1, y1, x2, y2]
                target[k] = target[k][:num_detections, :]
                target[k] = u.convert_boxes_to_x1y1x2y2(target[k])
    return targets

def draw_boxes(img, flight_id, frame_id, pred_boxes, true_boxes, outdir='outputs'):
    # unprocess img back to cv2 format
    img_np = img.permute(1, 2, 0).detach().cpu().numpy() * 255
    # convert rgb to bgr for cv2
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # slice predicted boxes for confidences
    pred_boxes = pred_boxes[:4].detach().cpu().numpy()
    true_boxes = true_boxes[:4].detach().cpu().numpy()
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (int(x1), int(y1), int(x2), int(y2)), (0, 0, 255), 2)
    for box in true_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'img_{flight_id}_{frame_id}.png')
    cv2.imwrite(outpath, img_np)

def save_model_weights(model, epoch, save_dir='model_weights'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_batch=None):
    model.train()
    running_loss = 0.0
    num_images = 0
    data_iter = tqdm(train_loader, desc="Training", leave=False)
    for i, (images, targets) in enumerate(data_iter):

        # process images and targets
        images = process_images(images, device)
        targets = process_targets(targets, device)

        # forwards with mixed precision
        with autocast():
            loss_dict, outputs = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # backwards
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        data_iter.set_postfix(loss=losses)
        torch.cuda.empty_cache()

        # update accumulators
        running_loss += losses.item()
        num_images += len(images)

        # break train loop at max batch
        if max_batch is not None and i == max_batch:
            break
    
    scheduler.step()
    # average running loss by number of images to calculate epoch loss
    epoch_loss = running_loss / num_images
    return epoch_loss

def val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type):
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = process_images(images, device)
            targets = process_targets(targets, device)

            with autocast():
                outputs = model(images, targets)
            breakpoint()

            for j, output in enumerate(outputs):
                print(f"Output: {output}")
                flight_id = targets[j]['flight_id']
                frame_id = targets[j]['frame_id']
                true_boxes = targets[j]['boxes']
                pred_boxes = output['boxes']
                draw_boxes(images[j], flight_id, frame_id, pred_boxes, true_boxes)


if __name__ == '__main__':

    model_dict = m.get_model_dict("FCOS", 7, 7)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_dict['model'].to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    loss_weights = model_dict['loss_weights']
    iou_thresh = -0.50
    conf_thresh = 0.00
    iou_type = "ciou"

    batch_size = 2
    train_loader, val_loader = prepare_data(batch_size)
    train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, 5)
    val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_thresh)


