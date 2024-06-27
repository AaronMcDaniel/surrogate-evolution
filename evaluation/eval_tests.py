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
from torch.cuda.amp import autocast

def dummy_prepare_data(batch_size):
    train_dataset = data.AOTDataset('train')
    val_dataset = data.AOTDataset('val')
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=data.my_collate)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=data.my_collate)
    # collate_fn=lambda x: tuple(zip(*x))
    return train_loader, val_loader


# takes in cv2 fromatted image tensor, full image path, true boxes tensor, predicted boxes tensor, flight id and frame id
def draw_bboxes(cv2_img, full_path, true_boxes, pred_boxes, flight_id, frame_id, outdir):
    cv2_img = np.array(cv2_img)
    for bbox in true_boxes:
        x, y, w, h = bbox[:4].detach().cpu().numpy()
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int (y + h)
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for bbox in pred_boxes:
        x, y, w, h = bbox[:4].detach().cpu().numpy()
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int (y + h)
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    os.makedirs(outdir, exist_ok=True)    
    save_path = os.path.join(outdir, f"img_{flight_id}_{frame_id}.png")
    cv2.imwrite(save_path, cv2_img)

def test_get_model():
    model_dict = m.get_model_dict(8)
    model = model_dict['model']
    print(model)
    summary.detection_model_summary(model, (3, 2048, 2448))

def get_optimizer(params, model_dict):
    default_params = {
        'SGD': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0, 'dampening': 0, 'nesterov': False},
        'Adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0},
        'Adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0, 'eps': 1e-10},
        'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False},
        'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01, 'amsgrad': False},
        'SparseAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
        'Adamax': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
        'ASGD': {'lr': 0.01, 'lambd': 1e-4, 'alpha': 0.75, 't0': 1e6, 'weight_decay': 0},
        'LBFGS': {'lr': 1.0, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-5, 'tolerance_change': 1e-9, 'history_size': 100, 'line_search_fn': None},
        'NAdam': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0, 'momentum_decay': 0.004},
        'RAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
        'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0, 'centered': False},
        'Rprop': {'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-6, 50)},
    }
    optimizer_type = model_dict.get('optimizer', 'SGD')
    optimizer_class_map = {
        'SGD': optim.SGD,
        'Adadelta': optim.Adadelta,
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'SparseAdam': optim.SparseAdam,
        'Adamax': optim.Adamax,
        'Adagrad': optim.Adagrad,
        'ASGD': optim.ASGD,
        'LBFGS': optim.LBFGS,
        'NAdam': optim.NAdam,
        'RMSprop': optim.RMSprop,
        'Rprop': optim.Rprop
    }
    optimizer_class = optimizer_class_map.get(optimizer_type)
    optimizer_defaults = default_params.get(optimizer_type, {})
    optimizer_params = {k: model_dict.get(f'optimizer_{k}', v) for k, v in optimizer_defaults.items()}
    sig = inspect.signature(optimizer_class)
    valid_params = {k: v for k, v in optimizer_params.items() if k in sig.parameters}
    valid_params['params'] = params
    optimizer = optimizer_class(**valid_params)
    return optimizer

def test_get_sched_optim():
    model_dict = m.get_model_dict(8)
    model = model_dict['model']
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(params, model_dict)
    scheduler = get_scheduler(optimizer, model_dict, 10, 2)
    print(f"Scheduler success: {scheduler.state_dict()}")
    print(f"Optimizer success: {optimizer.state_dict()}")

def get_scheduler(optimizer, model_dict, num_epochs, batch_size):
    default_params = {
        'StepLR': {'step_size': 30, 'gamma': 0.1, 'last_epoch': -1},
        'MultiStepLR': {'milestones': [30, 80], 'gamma': 0.1, 'last_epoch': -1},
        'ExponentialLR': {'gamma': 0.1, 'last_epoch': -1},
        'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.1, 'patience': 10, 'threshold': 0.0001, 'cooldown': 0, 'min_lr': 0, 'eps': 1e-08},
        'CosineAnnealingLR': {'T_max': 50, 'eta_min': 0, 'last_epoch': -1},
        'CosineAnnealingWarmRestarts': {'T_0': 10, 'T_mult': 2, 'eta_min': 0, 'last_epoch': -1},
        'OneCycleLR': {'max_lr': 0.1, 'total_steps': None, 'epochs': num_epochs, 'steps_per_epoch': batch_size, 'pct_start': 0.3, 'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85, 'max_momentum': 0.95, 'div_factor': 25.0, 'final_div_factor': 1e4, 'three_phase': False, 'last_epoch': -1, 'verbose': False},
        'ConstantLR': {'factor': 1.0, 'total_iters': 5},
        'MultiplicativeLR': {'lr_lambda': lambda epoch: 0.95, 'last_epoch': -1},
        'LambdaLR': {'lr_lambda': lambda epoch: 1, 'last_epoch': -1},
        'LinearLR': {'start_factor': 0.1, 'end_factor': 1.0, 'total_iters': 5, 'last_epoch': -1},
        'PolynomialLR': {'max_lr': 0.1, 'total_steps': 10, 'power': 1.0, 'last_epoch': -1},
        'ChainedScheduler': {'schedulers': [], 'last_epoch': -1},
        'CyclicLR': {'base_lr': 0.001, 'max_lr': 0.1, 'step_size_up': 2000, 'step_size_down': None, 'mode': 'triangular', 'gamma': 1.0, 'scale_fn': None, 'scale_mode': 'cycle', 'cycle_momentum': True, 'base_momentum': 0.8, 'max_momentum': 0.9, 'last_epoch': -1},
        'SequentialLR': {'schedulers': [], 'milestones': []},
    }
    scheduler_type = model_dict.get('lr_scheduler', 'StepLR')
    scheduler_class_map = {
        'StepLR': lr_scheduler.StepLR,
        'MultiStepLR': lr_scheduler.MultiStepLR,
        'ExponentialLR': lr_scheduler.ExponentialLR,
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'OneCycleLR': lr_scheduler.OneCycleLR,
        'ConstantLR': lr_scheduler.ConstantLR,
        'MultiplicativeLR': lr_scheduler.MultiplicativeLR,
        'LambdaLR': lr_scheduler.LambdaLR,
        'LinearLR': lr_scheduler.LinearLR,
        'PolynomialLR': lr_scheduler.PolynomialLR,
        'ChainedScheduler': lr_scheduler.ChainedScheduler,
        'CyclicLR': lr_scheduler.CyclicLR,
        'SequentialLR': lr_scheduler.SequentialLR
    }
    scheduler_class = scheduler_class_map.get(scheduler_type)
    scheduler_defaults = default_params.get(scheduler_type)
    scheduler_params = {k: model_dict.get(f'scheduler_{k}', v) for k, v in scheduler_defaults.items()}
    sig = inspect.signature(scheduler_class)
    valid_params = {k: v for k, v in scheduler_params.items() if k in sig.parameters}
    valid_params['optimizer'] = optimizer
    scheduler = scheduler_class(**valid_params)
    return scheduler

def process_image(img):
    img = (img.float() / 255.0)
    img = img.permute(2, 0, 1)
    return img

def create_metrics_df():
    return pd.DataFrame(columns=['epoch_num', 'train_epoch_loss', 'val_epoch_loss', 
                                       'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss',
                                       'center_loss', 'size_loss', 'obj_loss', 'precision', 
                                       'recall', 'f1_score', 'average_precision', 
                                       'true_positives', 'false_positives', 'false_negatives'])

# saves all model metrics and predictions to disc
def store_data(metrics_df: pd.DataFrame, save_path):
    outdir = 'metrics'
    save_path = os.path.join(outdir, save_path)
    os.makedirs(outdir, exist_ok=True)
    metrics_df.to_csv(save_path, index=False)

def dummy_engine():
    save_path = "test01"
    model_dict = m.get_model_dict(7)
    model = model_dict['model']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    batch_size = 2
    loss_weights = model_dict['loss_weights']
    iou_thresh = -0.50
    conf_thresh = 0.00
    iou_type = "ciou"
    num_epochs = 2
    optimizer = get_optimizer(params, model_dict)
    scheduler = get_scheduler(optimizer, model_dict, num_epochs, batch_size)
    train_loader, val_loader = dummy_prepare_data(batch_size)
    # all_preds = []
    metrics_df = create_metrics_df()
    print(metrics_df)
    for epoch in range(1, num_epochs + 1):
        # epoch_preds = {}
        # train_epoch_loss = dummy_train_one_epoch(model, device, train_loader, optimizer, scheduler)
        # epoch_metrics = dummy_val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type)
        epoch_metrics = {}
        epoch_metrics['epoch_num'] = epoch
        # epoch_metrics['train_epoch_loss'] = train_epoch_loss
        metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
        # all_preds[epoch - 1] = epoch_preds
        store_data(metrics_df, save_path)
    print(f"{num_epochs} of training and validation completed")
        


def dummy_train_one_epoch(model, device, train_loader, optimizer, scheduler):
    model.train()
    train_epoch_loss = 0.0
    num_images = 0

    data_iter = tqdm(train_loader, desc="Training", leave=False)
    for i, (images, targets) in enumerate(data_iter):

        if (i == 200):
            print(f"{i} batches of data trained successfully!")
            break

        images = [(process_image(img)).to(device) for img in images]
        num_images += len(images)

        for t in targets:
                num_detections = t['num_detections']
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)
                    if k == 'boxes':
                        # slices through empty true boxes and convert to [x1, y1, x2, y2]
                        t[k] = t[k][:num_detections, :]
                        t[k] = u.convert_boxes_to_x1y1x2y2(t[k])

        optimizer.zero_grad()

        # forward
        loss_dict = model(images, targets)
        # print(f"Model loss dictionary: {loss_dict}")
        losses = sum(loss for loss in loss_dict.values())
        # print(f"Summed model losses: {losses.item()}")
        train_epoch_loss += losses.item()
        
        model.train()
        # backward
        losses.backward()
        data_iter.set_postfix(loss=losses.item())
        optimizer.step()
    
    # step scheduler
    scheduler.step()
    train_epoch_loss /= num_images
    return train_epoch_loss

def dummy_val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type):
    model.eval()
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

    with torch.no_grad():
        data_iter = tqdm(val_loader)
        for i, (images, targets) in enumerate(data_iter):
            if (i == 10):
                print(f"{i} batches validated successfully")
                break
            os.popen('rm -rf loss_matches_on_img')
            os.popen('rm -rf matches_on_img')

            cv2_images = images
            images = [(process_image(img)).to(device) for img in images]

            for t in targets:
                num_detections = t['num_detections']
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)
                    if k == 'boxes':
                        # slices through empty true boxes
                        t[k] = t[k][:num_detections, :]

            outputs = model(images)

            for j, output in enumerate(outputs):
                flight_id = str(targets[j]['flight_id'])
                # used to have item()
                frame_id = int(targets[j]['frame_id'])
                true_boxes = targets[j]['boxes']

                # access predicted boxes, convert them to [x, y, w, h] and concatenate scores
                pred_boxes = output['boxes']
                pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
                scores = output['scores']
                # print(f"scores: {scores}")
                pred_boxes = u.cat_scores(pred_boxes, scores)
                # print(f"pred boxes after concat: {pred_boxes}")

                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)
                # print(f"matched bboxes: {matches}")

                # update accumulators
                num_tp = len(matches)
                num_fp = len(fp)
                num_fn = len(fn)
                total_tp += num_tp
                total_fp += num_fp
                total_fn += num_fn
                num_preds += len(pred_boxes)
                num_labels += len(true_boxes)

                # draw matched boxes on the image
                matched_pred_boxes = torch.zeros((len(matches), 5))
                matched_true_boxes = torch.zeros((len(matches), 4))
                for idx, (t, (p, _)) in enumerate(matches.items()):
                    matched_true_boxes[idx] = t
                    matched_pred_boxes[idx] = p
                full_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/' + targets[j]['path']
                cv2_img = cv2_images[j]
                outdir_1 = 'matches_on_img'
                draw_bboxes(cv2_img, full_path, matched_true_boxes, matched_pred_boxes[:, :4], flight_id, frame_id, outdir_1)

                # match loss boxes and draw loss-matched boxes on image
                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
                matched_loss_pred_boxes = torch.zeros((len(loss_matches), 5))
                matched_loss_true_boxes = torch.zeros((len(loss_matches), 4))
                for idx, (t, (p, _)) in enumerate(loss_matches.items()):
                    matched_loss_true_boxes[idx] = t
                    matched_loss_pred_boxes[idx] = p
                
                outdir_2 = 'loss_matches_on_img'
                draw_bboxes(cv2_img, full_path, matched_loss_true_boxes, matched_loss_pred_boxes[:, :4], flight_id, frame_id, outdir_2)

                # get loss tensor
                loss_tensor = c.compute_weighted_loss(loss_matches, loss_weights, iou_type)

                # update more accumulators
                val_image_loss = loss_tensor[7]
                print(f"loss: {loss_tensor[7].item()}")
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

    val_epoch_loss /= num_preds
    iou_loss /= num_preds
    giou_loss /= num_preds
    diou_loss /= num_preds
    ciou_loss/= num_preds
    center_loss/= num_preds
    size_loss /= num_preds
    obj_loss /= num_preds
    epoch_f1, epoch_pre, epoch_rec = u.f1_score(total_tp, total_fn, total_fp)
    pre_curve, rec_curve = u.precision_recall_curve(confidences, confusion_status, num_labels)
    epoch_avg_pre = u.AP(pre_curve, rec_curve)
    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre)

    return {
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

    #     for k, v in targets.items():
    #         if isinstance(v, torch.Tensor):
    #             targets[k] = v.to(device)
    #         elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
    #             targets[k] = [vi.to(device) for vi in v]

    #     outputs = model(images)
    #     # print(f"Model inferences: {outputs}")
    #     with torch.no_grad():
    #         for j, img in enumerate(images):
    #             flight_id = str(targets['flight_id'][j])
    #             frame_id = int(targets['frame_id'][j].item())
    #             print(f"Flight id, frame id: ({flight_id, frame_id})")

    #             # slice through true boxes based on number of detections
    #             true_num_objects = targets['num_detections'][j]
    #             true_boxes = targets['bboxes'][j]
    #             true_boxes = true_boxes[:true_num_objects, :]

    #             # concatenate scores on the end of predicted boxes tensor
    #             pred_boxes = outputs[j]['boxes']
    #             pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
    #             scores = outputs[j]['scores']
    #             pred_boxes = u.cat_scores(pred_boxes, scores)

# # draw bounding boxes 
# full_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/' + targets['path'][j]
# cv2_img = cv2_images[j]
# outdir = 'all_preds_on_img'
# draw_bboxes(cv2_img, full_path, true_boxes, pred_boxes[:, :4], flight_id, frame_id, outdir)

# # test different NMS
# nms1_pred_boxes = u.non_max_suppresion(pred_boxes, iou_thresh, 1, "ciou")
# post_NMS_outdir = 'nms1_preds_on_img'
# draw_bboxes(cv2_img, full_path, true_boxes, nms1_pred_boxes[:, :4], flight_id, frame_id, post_NMS_outdir)

# nms2_pred_boxes = u.non_max_suppresion(pred_boxes, iou_thresh, 2, "ciou")
# post_NMS_outdir = 'nms2_preds_on_img'
# draw_bboxes(cv2_img, full_path, true_boxes, nms2_pred_boxes[:, :4], flight_id, frame_id, post_NMS_outdir)

# nms3_pred_boxes = u.non_max_suppresion(pred_boxes, iou_thresh, 3, "ciou")
# post_NMS_outdir = 'nms3_preds_on_img'
# draw_bboxes(cv2_img, full_path, true_boxes, nms3_pred_boxes[:, :4], flight_id, frame_id, post_NMS_outdir)

if __name__ == '__main__':
    model_dict = m.get_model_dict(7)
    model = model_dict['model']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    batch_size = 10
    loss_weights = model_dict['loss_weights']
    iou_thresh = -0.50
    conf_thresh = 0.00
    iou_type = "ciou"
    num_epochs = 2
    optimizer = get_optimizer(params, model_dict)
    scheduler = get_scheduler(optimizer, model_dict, num_epochs, batch_size)
    train_loader, val_loader = dummy_prepare_data(batch_size)
    train_epoch_loss = dummy_train_one_epoch(model, device, train_loader, optimizer, scheduler)
    print(f"train epoch loss: {train_epoch_loss}")
    epoch_metrics = dummy_val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type)
    print(f"validation epoch metrics: {epoch_metrics}")


