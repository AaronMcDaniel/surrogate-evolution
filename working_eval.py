import aot_dataset as data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import os
import sample_model as m
import model_summary as summary
import inspect
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import utils as u
import criterion as c


def prepare_data(train_seed, val_seed, batch_size=5):
    train_dataset = data.AOTDataset('train', seed=train_seed, string=1)
    val_dataset = data.AOTDataset('val', seed=val_seed, string=1)
    train_sampler = data.AOTSampler(train_dataset, batch_size, train_seed)
    val_sampler = data.AOTSampler(val_dataset, batch_size, val_seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data.my_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=data.my_collate, num_workers=4)
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

def draw_boxes(img, flight_id, frame_id, pred_boxes, true_boxes, outdir='images'):
    # unprocess img back to cv2 format
    img_np = img.permute(1, 2, 0).detach().cpu().numpy() * 255
    img_np = img_np.astype(np.uint8)
    # convert rgb to bgr for cv2
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # scale boxes back up to pixel values
    # NOTE scale_boxes makes a deep copy of the boxes so they are not modified in place
    scaled_pred_boxes = u.scale_boxes(pred_boxes)
    # convert to x1y1x2y2 format
    scaled_pred_boxes = u.convert_boxes_to_x1y1x2y2(scaled_pred_boxes)
    scaled_pred_boxes = scaled_pred_boxes[:4].detach().cpu().numpy()
    scaled_true_boxes = u.scale_boxes(true_boxes)
    scaled_true_boxes = u.convert_boxes_to_x1y1x2y2(scaled_true_boxes)
    scaled_true_boxes = scaled_true_boxes[:4].detach().cpu().numpy()
    for box in scaled_pred_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for box in scaled_true_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'img_{flight_id}_{frame_id}.png')
    cv2.imwrite(outpath, img_np)

def save_model_weights(model, model_type, num_images, save_dir='weights'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_{num_images}_img.pth')
    torch.save(model.state_dict(), save_path)

def load_model_weights(model, device, file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'The directory path {file_path} does not exist.')
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def create_metrics_df():
    return pd.DataFrame(columns=['epoch_num', 'train_epoch_loss', 'val_epoch_loss', 
                                       'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss',
                                       'center_loss', 'size_loss', 'obj_loss', 'precision', 
                                       'recall', 'f1_score', 'average_precision', 
                                       'true_positives', 'false_positives', 'false_negatives'])

def inference_on_img(model, device, flight_id, frame_id, val_dataset):
    image, target = val_dataset.find_image(frame_id, flight_id)
    model.eval()
    with torch.no_grad():
        image = image.permute(2, 0, 1).float()
        image = image / 255.0
        image = image.to(device)
        true_boxes = target['boxes'].to(device)
        num_detections = target['num_detections']
        true_boxes = true_boxes[:num_detections, :]
        with torch.autocast(device_type=device.type):
            output = model([image])
        # extract predicted boxes and scores
        pred_boxes = output[0]['boxes']
        pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
        true_boxes = u.convert_boxes_to_xywh(true_boxes)
        scores = output[0]['scores']
        pred_boxes = u.cat_scores(pred_boxes, scores)
        iou_thresh = -0.50
        conf_thresh = 0.00
        iou_type = "ciou"
        matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)
    print("Successful inference!")

def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_batch=None):
    model.train()
    running_loss = 0.0
    num_images = 0
    data_iter = tqdm(train_loader, desc="Training")

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
            save_model_weights(model, 'RetinaNet', 25000)
            break
    
    scheduler.step()
    # average running loss by number of images to calculate epoch loss
    epoch_loss = running_loss / num_images
    return epoch_loss

def val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, max_batch=None):
    os.popen('rm -rf matches')
    os.popen('rm -rf plots')
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    model.eval()
    data_iter = tqdm(val_loader, desc="Evaluating epoch")
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_iter):
            os.popen('rm -rf images')
            images = process_images(images, device)
            targets = process_targets(targets, device)

            with autocast():
                outputs = model(images, targets)

            loss_matrix = torch.tensor(batch_size, len(loss_weights) + 1, dtype=torch.float32)

            for j, image in enumerate(images):
                flight_id = targets[j]['flight_id']
                frame_id = targets[j]['frame_id']
                true_boxes = u.norm_box_scale(targets[j]['boxes'])
                pred_boxes = u.norm_box_scale(outputs[j]['boxes'])
                pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
                true_boxes = u.convert_boxes_to_xywh(true_boxes)
                # NOTE pred_boxes and true_boxes are currently normalized dbetween [0, 1] and in xywh format
                draw_boxes(image, flight_id, frame_id, pred_boxes, true_boxes)
                scores = outputs[j]['scores']
                pred_boxes = u.cat_scores(pred_boxes, scores)
                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)
                num_tp = len(matches)
                num_fp = len(fp)
                num_fn = len(fn)
                total_tp += num_tp
                total_fp += num_fp
                total_fn += num_fn
                num_preds += len(pred_boxes)
                num_labels += len(true_boxes)
                matched_pred_boxes = torch.zeros((len(matches), 4))
                matched_true_boxes = torch.zeros((len(matches), 4))
                for idx, (t, (p, _)) in enumerate(matches.items()):
                    matched_true_boxes[idx] = t
                    matched_pred_boxes[idx] = p[:4]
                draw_boxes(image, flight_id, frame_id, matched_pred_boxes, matched_true_boxes, outdir='matches')
                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
                loss_tensor = c.compute_weighted_loss(loss_matches, loss_weights, iou_type)
                loss_matrix[j]
                val_image_loss = loss_tensor[7]
                val_epoch_loss += val_image_loss
                iou_loss += loss_tensor[0]
                giou_loss += loss_tensor[1]
                diou_loss += loss_tensor[2]
                ciou_loss += loss_tensor[3]
                center_loss += loss_tensor[4]
                size_loss += loss_tensor[5]
                obj_loss += loss_tensor[6]
                for _, (true_pos, _) in matches.items():
                    confidences.append(true_pos[4].item())
                    confusion_status.append(True)
                for false_pos in fp:
                    confidences.append(false_pos[4].item())
                    confusion_status.append(False)
                    
            # break val loop at max batch
            if max_batch is not None and i == max_batch:
                break
    breakpoint()
    val_epoch_loss /= (num_preds +  1e-9)
    iou_loss /= (num_preds  + 1e-9)
    giou_loss /= (num_preds + 1e-9)
    diou_loss /= (num_preds + 1e-9)
    ciou_loss/= (num_preds + 1e-9)
    center_loss/= (num_preds + 1e-9)
    size_loss /= (num_preds + 1e-9)
    obj_loss /= (num_preds + 1e-9)
    epoch_f1, epoch_pre, epoch_rec = u.f1_score(total_tp, total_fn, total_fp)
    pre_curve, rec_curve = u.precision_recall_curve(confidences, confusion_status, num_labels)
    epoch_avg_pre = u.AP(pre_curve, rec_curve)
    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre)

    epoch_metrics = {
        'val_epoch_loss': val_epoch_loss,
        'iou_loss':iou_loss,
        'giou_loss': giou_loss,
        'diou_loss': diou_loss,
        'ciou_loss': ciou_loss,
        'center_loss': center_loss,
        'size_loss': size_loss,
        'obj_loss': obj_loss,
        'precision': epoch_pre,
        'recall': epoch_rec,
        'f1_score': epoch_f1,
        'average_precision': epoch_avg_pre,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }
    return epoch_metrics

def custom_train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, custom_loss, max_batch=None):
    model.train()
    train_epoch_loss = 0.0
    num_preds = 0
    data_iter = tqdm(train_loader, desc="Training")

    for i, (images, targets) in enumerate(data_iter):

        # process images and targets
        images = process_images(images, device)
        targets = process_targets(targets, device)

        # forwards with mixed precision
        with autocast():
            loss_dict, outputs = model(images, targets)
            if custom_loss:
                losses = torch.zeros(1, requires_grad=True, device=device)
            else:
                losses = sum(loss for loss in loss_dict.values())

        for j, image in enumerate(images):
            num_preds += len(outputs[j]['boxes'])
            if custom_loss:
                true_boxes = u.norm_box_scale(targets[j]['boxes'])
                pred_boxes = u.norm_box_scale(outputs[j]['boxes'])
                pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
                true_boxes = u.convert_boxes_to_xywh(true_boxes)
                # NOTE pred_boxes and true_boxes are currently normalized between [0, 1] and in xywh format
                scores = outputs[j]['scores']
                pred_boxes = u.cat_scores(pred_boxes, scores) 
                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", "ciou")
                size_loss = c.size_loss(loss_matches)
                center_loss = c.center_loss(loss_matches)
                total_loss = size_loss * 0.6 + center_loss * 0.4
                # loss_tensor = c.compute_weighted_loss(loss_matches, loss_weights, iou_type)
                losses = losses + total_loss

        # backwards
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        data_iter.set_postfix(loss=losses)
        torch.cuda.empty_cache()

        # update accumulators
        train_epoch_loss += losses.item()

        # break train loop at max batch
        if max_batch is not None and i == max_batch:
            save_model_weights(model, 'RetinaNet_Loss_Testing', 5)
            break
    
    scheduler.step()
    # average running loss by number of images to calculate epoch loss
    train_epoch_loss /= (num_preds + 1e-9)
    return train_epoch_loss
        
if __name__ == '__main__':
    model_dict = m.get_model_dict("RetinaNet", 7, 7)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_dict['model'].to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()
    train_seed=0
    val_seed=0

    loss_weights = model_dict['loss_weights']
    custom_loss = True
    iou_thresh = 0.2
    conf_thresh = 0.00
    iou_type = "ciou"

    batch_size = 5
    train_loader, val_loader = prepare_data(train_seed, val_seed, batch_size)
    
    # train_epoch_loss = custom_train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, custom_loss, max_batch=50)
    # print(f'Train Loss: {train_epoch_loss}')

    model = load_model_weights(model, device, '/home/tthakur9/precog-opt-grip/weights/RetinaNet_25000_img.pth')
    epoch_metrics = val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, max_batch=1)
    print("Evaluation finished!")
    print(f"Metrics: {epoch_metrics}")

    
