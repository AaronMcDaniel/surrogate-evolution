"""
Uses saved model predictions to re-evaluate a single individual.
Overwrites old metrics.csv file and PR-curve plot.
Normally sbatched by launcher script re_eval.py
"""


import argparse
import itertools
import os
import pickle
import toml
import numpy
import torch
import tqdm
import aot_dataset as data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import os
import utils as u
import criterion as c
from eval import process_images, process_targets, process_preds_truths, create_metrics_df


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outdir', required=True)
parser.add_argument('-g', '--genomedir', required=True)
args = parser.parse_args()
outdir = args.outdir
genome_folder = args.genomedir

genome_hash = os.path.basename(genome_folder)
print('Genome hash: ', genome_hash)
print('--------------------')
predictions_path = os.path.join(genome_folder, 'predictions.pkl')

# load config attributes
configs = toml.load(os.path.join(outdir, "conf.toml"))
pipeline_config = configs["pipeline"]
model_config = configs["model"]
codec_config = configs["codec"]
data_config = configs["data"]
cfg = pipeline_config | model_config | codec_config | data_config


def prepare_data(cfg, val_seed, batch_size=5):
    cache_thresh = cfg['cache_thresh']
    max_size = None
    try:
        max_size = cfg['max_size']
    except KeyError:
        pass 
    val_dataset = data.AOTDataset('val', seed=val_seed, string=1, cache_thresh=cache_thresh, max_size=max_size)
    val_sampler = data.AOTSampler(val_dataset, batch_size, val_seed)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=data.my_collate, num_workers=4)
    return val_loader

# saves all model metrics and predictions to disc
def store_data(metrics_df: pd.DataFrame, dir):
    metrics_out = f'{dir}/metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

    metrics_df.to_csv(metrics_out, index=False)

def val_one_epoch(predictions, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_num, folder, max_batch=None):
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        val_loader = itertools.islice(val_loader, max_batch)
    data_iter = tqdm(val_loader, desc="Evaluating", total=max_batch)
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_iter):
            images = process_images(images, device)
            targets = process_targets(targets, device)
            outputs = predictions[epoch_num]

            val_batch_loss = torch.zeros(1, dtype=torch.float32, device=device)
            for j, image in enumerate(images):
                pred_boxes, true_boxes, flight_id, frame_id = process_preds_truths(targets[j], outputs[(targets[j]['flight_id'], targets[j]['frame_id'])])
                # NOTE pred boxes are normalized, in xywh format, and have scores, and true boxes are nomalized and in xywh format

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
    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre, folder)

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



# get predictions
pred_file = open(predictions_path, 'rb')
predictions = pickle.load(pred_file)
pred_file.close()

metrics_df = create_metrics_df()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# retrieve config attributes
num_loss_comp = cfg['num_loss_components']
batch_size = cfg['batch_size']
batches_per_epoch = cfg['batches_per_epoch']
iou_thresh = cfg['iou_thresh']
conf_thresh = cfg['conf_thresh']
iou_type = cfg['iou_type']
val_seed = cfg['val_seed']
best_epoch_criteria = cfg['best_epoch_criteria']

val_loader = prepare_data(cfg, val_seed, batch_size)
loss_weights = torch.full((num_loss_comp, ), 1.0 / num_loss_comp)

# iterate through val dataset to get targets
for epoch in range(len(predictions)):
    epoch_metrics = val_one_epoch(predictions, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch, genome_folder, max_batch=batches_per_epoch)
    epoch_metrics['epoch_num'] = epoch + 1
    epoch_metrics['train_epoch_loss'] = 0
    epoch_metrics_df = pd.DataFrame([epoch_metrics])
    metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
    # save metrics_df back to disc
    store_data(metrics_df, genome_folder)
    print(f'Epoch {epoch + 1} done!')

print('Re-eval done!')