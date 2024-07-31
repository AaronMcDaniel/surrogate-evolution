import argparse
import itertools
import os
import pickle
import toml
import numpy
import torch
import tqdm
import sys
sys.path.insert(0, '/home/eharpster3/precog-opt-grip')
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
parser.add_argument('--job_id', type=int, default=None, help='num of job in array')
args = parser.parse_args()
outdir = args.outdir
genome_folder = args.genomedir
job_id = args.job_id
if job_id == 11:
    genome_folder = 'dmytro_metrics/complex'
genome_hash = os.path.basename(genome_folder)
print('Genome hash: ', genome_hash)
print('--------------------')

if job_id is not None:
    if job_id < 10:
        predictions_path = 'dmytro_repo/data/results/run0/result0' + str(job_id) + '.pkl'
    elif job_id == 10:
        predictions_path = 'dmytro_repo/data/results/run0/result' + str(job_id) + '.pkl'
    else:
        predictions_path = 'dmytro_repo/data/results/run0/result.pkl'
#os.path.join(genome_folder, 'predictions.pkl')

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

# converts tagerts from data loader to pytorch acceptable format
def process_targets(target, device):
    num_detections = target['num_detections']
    for k, v in target.items():
        # move to device if tensor
        if isinstance(v, torch.Tensor):
            target[k] = v.to(device)
        if k == 'boxes':
            # slice off empty boxes and convert to [x1, y1, x2, y2]
            target[k] = target[k][:num_detections, :]
            target[k] = u.convert_boxes_to_x1y1x2y2(target[k])
    return target

def val_one_epoch(predictions, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_num, folder, max_batch=None, labels_cache=None):
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        val_loader = itertools.islice(val_loader, max_batch)
    #data_iter = tqdm(val_loader, desc="Evaluating", total=max_batch)
    with torch.no_grad():
        #print(len(predictions.items()))
        for i, (key, val) in enumerate(tqdm(predictions.items())):
            #images = process_images(images, device)
            flight_id, frame_id = key
            target = labels_cache[flight_id][frame_id]
            target = process_targets(target, device)
            outputs = predictions

            val_batch_loss = torch.zeros(1, dtype=torch.float32, device=device)
            #for j, image in enumerate(images):
            val['boxes'] = torch.tensor(val['boxes'], device=device, dtype=torch.float32)
            val['scores'] = torch.tensor(val['scores'], device=device, dtype=torch.float32)
            val['labels'] = torch.tensor(val['labels'], device=device, dtype=torch.float32)

            pred_boxes, true_boxes, flight_id, frame_id = process_preds_truths(target, val) #outputs[(target['flight_id'], target['frame_id'])])
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

            curve_matches, curve_fp, _ = u.match_boxes(pred_boxes, true_boxes, iou_thresh, 0.00, 'val', iou_type)

            for _, (true_pos, _) in curve_matches.items():
                confidences.append(true_pos[4].item())
                confusion_status.append(True)
            for false_pos in curve_fp:
                confidences.append(false_pos[4].item())
                confusion_status.append(False)
            # for _, (true_pos, _) in matches.items():
            #     confidences.append(true_pos[4].item())
            #     confusion_status.append(True)
            # for false_pos in fp:
            #     confidences.append(false_pos[4].item())
            #     confusion_status.append(False)

            #data_iter.set_postfix(loss=val_batch_loss)

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
    
    pr_curve_metrics = {}
    pr_curve_metrics['pre_curve'] = pre_curve
    pr_curve_metrics['rec_curve'] = rec_curve
    pr_curve_metrics['epoch_avg_pre'] = epoch_avg_pre
    base_path = 'dmytro_metrics'
    print(job_id)
    if job_id < 11:
        path = base_path + '/epochs/' + str(job_id)
    elif job_id == 11:
        path = base_path + '/complex'
    else:
        path = base_path + '/' + str(job_id)
    with open(path + '/pr_curve_metrics.pkl', 'wb') as f:
        pickle.dump(pr_curve_metrics, f)

    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre, path)

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

with open('/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/everything.pkl', 'rb') as file:
    labels_cache = pickle.load(file)

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
#for epoch in range(len(predictions)):
epoch_metrics = val_one_epoch(predictions, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, 1 - 1, genome_folder, max_batch=batches_per_epoch, labels_cache=labels_cache)
epoch_metrics['epoch_num'] = job_id
epoch_metrics['train_epoch_loss'] = 0
epoch_metrics_df = pd.DataFrame([epoch_metrics])
metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
# save metrics_df back to disc
store_data(metrics_df, genome_folder)
print(f'Epoch {job_id} done!')

print('Re-eval done!')