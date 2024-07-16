import itertools

import toml
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
import eval as e
from codec import Codec
from itertools import islice

# helper method to visualize predictions
def draw_boxes(img, flight_id, frame_id, pred_boxes, true_boxes, outdir='images'):
    # unprocess img 
    img_np = img.permute(1, 2, 0).detach().cpu().numpy() * 255
    img_np = img_np.astype(np.uint8)
    # convert rgb to bgr for cv2
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # scale boxes back up to pixel values
    # NOTE scale_boxes makes a deep copy of the boxes so they are not modified in place
    scaled_pred_boxes = u.scale_boxes(pred_boxes)
    # convert to x1y1x2y2 format and convert to numpy arr
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


# saves model weights for testing
def save_model_weights(model, model_type, num_images, save_dir='weights'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_{num_images}_img.pth')
    torch.save(model.state_dict(), save_path)


# loads model weights for testing
def load_model_weights(model, device, file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'The directory path {file_path} does not exist.')
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def find_img_zero_det(val_seed, batch_size):
    val_dataset = data.AOTDataset('val', seed=val_seed, string=1, max_size=None)
    val_sampler = data.AOTSampler(val_dataset, batch_size, seed=val_seed)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=data.my_collate, num_workers=4)
    data_iter = tqdm(val_loader)
    for _, targets in data_iter:
        for t in targets:
            if t['num_detections'] == 0:
                return t['flight_id'], t['frame_id']


def inference_on_img(model, device, flight_id, frame_id, val_dataset, loss_weights):
    image, target = val_dataset.find_image(frame_id, flight_id)
    model.eval()
    with torch.no_grad():
        image = image.permute(2, 0, 1).float()
        image = image / 255.0
        image = image.to(device)
        true_boxes = target['boxes'].to(device)
        num_detections = target['num_detections']
        true_boxes = true_boxes[:num_detections, :]
        true_boxes = u.convert_boxes_to_x1y1x2y2(true_boxes)

        with torch.autocast(device_type=device.type):
            output = model([image])

        pred_boxes = u.norm_box_scale(output[0]['boxes'])
        true_boxes = u.norm_box_scale(true_boxes)
        pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
        true_boxes = u.convert_boxes_to_xywh(true_boxes)
        pred_boxes = u.clean_zero_dim(pred_boxes)
        scores = output[0]['scores']
        pred_boxes = u.cat_scores(pred_boxes, scores)
        loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, 'train', 'ciou')
        loss_tensor = c.compute_weighted_loss(loss_matches, pred_boxes, true_boxes, loss_weights, 'ciou')
        print(loss_tensor)


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_batch=None):
    model.train()
    running_loss = 0.0
    num_images = 0
    data_iter = tqdm(train_loader, desc="Training")

    for i, (images, targets) in enumerate(data_iter):

        # process images and targets
        images = e.process_images(images, device)
        targets = e.process_targets(targets, device)

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
            e.save_model_weights(model, 'RetinaNet', 25000)
            break
    
    scheduler.step()
    # average running loss by number of images to calculate epoch loss
    epoch_loss = running_loss / num_images
    return epoch_loss


def custom_train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, custom_loss, loss_weights, iou_type, max_batch=None):
    model.train()
    train_epoch_loss = 0.0
    num_preds = 0
    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        train_loader = itertools.islice(train_loader, max_batch)
    data_iter = tqdm(train_loader, desc="Training", total=max_batch)

    for i, (images, targets) in enumerate(data_iter):

        # process images and targets
        images = e.process_images(images, device)
        targets = e.process_targets(targets, device)

        # forwards with mixed precision
        with autocast():
            loss_dict, outputs = model(images, targets)
            if custom_loss:
                losses = torch.zeros(1, requires_grad=True, device=device, dtype=torch.float32)
            else:
                losses = sum(loss for loss in loss_dict.values())

            if custom_loss:
                for j, image in enumerate(images):
                    pred_boxes, true_boxes, flight_id, frame_id = e.process_preds_truths(targets[j], outputs[j])
                    
                    loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type=iou_type)
                    loss_tensor = c.compute_weighted_loss(loss_matches, pred_boxes, true_boxes, loss_weights, iou_type)
                    image_loss = loss_tensor[7]
                    losses = losses + image_loss

        # backwards
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        data_iter.set_postfix(loss=losses.item())
        torch.cuda.empty_cache()

        # update accumulators
        train_epoch_loss += losses.item()

    save_model_weights(model, model_type='genome', num_images=5000)
    train_epoch_loss = train_epoch_loss / max_batch if max_batch is not None else len(train_loader)
    e.step_scheduler(scheduler, train_epoch_loss)
    # average running loss by number of images to calculate epoch loss
    train_epoch_loss /= (num_preds + 1e-9)
    return train_epoch_loss


def val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, max_batch=None):
    os.popen('rm -rf images')
    os.popen('rm -rf plots')
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    model.eval()
    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        val_loader = itertools.islice(val_loader, max_batch)
    data_iter = tqdm(val_loader, desc="Evaluating", total=max_batch)
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_iter):
            images = e.process_images(images, device)
            targets = e.process_targets(targets, device)
            with autocast():
                outputs = model(images, targets)

            val_batch_loss = torch.zeros(1, dtype=torch.float32, device=device)
            for j, image in enumerate(images):
                pred_boxes, true_boxes, flight_id, frame_id = e.process_preds_truths(targets[j], outputs[j])
                # NOTE pred boxes are normalized, in xywh format, and have scores, and true boxes are nomalized and in xywh format
                draw_boxes(image, flight_id, frame_id, pred_boxes[:, :4], true_boxes, outdir='images')

                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)
                num_tp = len(matches)
                num_fp = len(fp)
                num_fn = len(fn)
                total_tp += num_tp
                total_fp += num_fp
                total_fn += num_fn
                num_preds += len(pred_boxes)
                num_labels += len(true_boxes)
                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
                loss_tensor = c.compute_weighted_loss(loss_matches, pred_boxes, true_boxes, loss_weights, iou_type)
                
                val_image_loss = loss_tensor[7]
                val_batch_loss += val_image_loss
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

            data_iter.set_postfix(loss=val_batch_loss)
            torch.cuda.empty_cache()

    val_epoch_loss /= (num_preds +  1e-9)
    iou_loss /= (num_preds  + 1e-9)
    giou_loss /= (num_preds + 1e-9)
    diou_loss /= (num_preds + 1e-9)
    ciou_loss/= (num_preds + 1e-9)
    center_loss/= (num_preds + 1e-9)
    size_loss /= (num_preds + 1e-9)
    obj_loss /= (num_preds + 1e-9)
    uw_val_epoch_loss = iou_loss + giou_loss + diou_loss + ciou_loss + center_loss + size_loss + obj_loss
    epoch_f1, epoch_pre, epoch_rec = u.f1_score(total_tp, total_fn, total_fp)
    pre_curve, rec_curve = u.precision_recall_curve(confidences, confusion_status, num_labels)
    epoch_avg_pre = u.AP(pre_curve, rec_curve)
    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre)

    epoch_metrics = {
        'uw_val_epoch_loss': uw_val_epoch_loss.item(),
        'val_epoch_loss': val_epoch_loss.item(),
        'iou_loss':iou_loss.item(),
        'giou_loss': giou_loss.item(),
        'diou_loss': diou_loss.item(),
        'ciou_loss': ciou_loss.item(),
        'center_loss': center_loss.item(),
        'size_loss': size_loss.item(),
        'obj_loss': obj_loss.item(),
        'precision': epoch_pre,
        'recall': epoch_rec,
        'f1_score': epoch_f1,
        'average_precision': epoch_avg_pre,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }
    return epoch_metrics

        
if __name__ == '__main__':
    iou_thresh = 0.0
    conf_thresh = 0.2
    iou_type = "ciou"
    train_seed = 0
    val_seed = 0
    batch_size = 1
    num_epochs = 1
    max_batch = 5000

    hash_path = '/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution/generation_1/c8ce86e998'
    split_list = hash_path.split('/')
    hash = split_list[-1]
    input_path = '/'.join(split_list[:-2]) + '/eval_inputs/' + f'eval_input_gen{split_list[-2].split("_")[-1]}.csv'
    input_df = pd.read_csv(input_path)
    genome_str = input_df.loc[input_df['hash'] == hash].to_dict('records')[0]['genome']

    codec = Codec(7)
    model_dict = codec.decode_genome(genome_str, 7)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_dict['model'].to(device)
    params = model.parameters()
    optimizer = e.get_optimizer(params, model_dict)
    scheduler = e.get_scheduler(optimizer, model_dict, num_epochs, batch_size)
    scaler = GradScaler()

    loss_weights = model_dict['loss_weights']
    configs = toml.load("conf.toml")
    model_config = configs["model"]
    codec_config = configs["codec"]
    data_config = configs["data"]
    all_config = model_config | codec_config | data_config
    train_loader, val_loader = e.prepare_data(all_config, train_seed, val_seed, batch_size)
    # model = load_model_weights(model, device, os.path.join(hash_path, 'last_epoch.pth'))
    model = load_model_weights(model, device, '/home/tthakur9/precog-opt-grip/weights/genome_100_img.pth')

    train_epoch_loss = custom_train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler,
                                              custom_loss=True, loss_weights=loss_weights, iou_type=iou_type, max_batch=max_batch)
    print(f"Custom train loss: {train_epoch_loss}")
    epoch_metrics = val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, max_batch=2)
    print("Evaluation finished!")
    print(f"Metrics: {epoch_metrics}")

    
