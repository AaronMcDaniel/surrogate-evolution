
def dummy_train_one_epoch(model, device, train_loader, optimizer, scheduler):
    # delete previous weights directory
    os.popen('rm -rf most_recent_train')
    model.train()
    train_epoch_loss = 0.0
    num_preds = 0

    data_iter = tqdm(train_loader, desc="Training", leave=False)
    for i, (images, targets) in enumerate(data_iter):

        # stop at i batches and save weights
        if (i == 2):
            model_weights_dir = 'most_recent_train'
            os.makedirs(model_weights_dir, exist_ok=True)
            model_weights_path = os.path.join(model_weights_dir, 'model_weights.pth')
            torch.save(model.state_dict(), model_weights_path)
            print(f"{i} batches of data trained successfully! Model weights saved to {model_weights_path}")
            break

        # ensures grey-scale values are between [0, 1] and in format [C, H, W]
        images = [(u.process_image(img)).to(device) for img in images]
        # move tensor targets to device
        for t in targets:
                num_detections = t['num_detections']
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)
                    if k == 'boxes':
                        # slices through empty true boxes and convert to [x1, y1, x2, y2]
                        t[k] = t[k][:num_detections, :]
                        t[k] = u.convert_boxes_to_x1y1x2y2(t[k])

        # zero gradients and get models's losses and predictions
        optimizer.zero_grad()
        loss_dict, outputs = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # iterate through each predcition
        for j, output in enumerate(outputs):

            # true boxes are currently in [x1, y1, x2, y2] format
            true_boxes = targets[j]['boxes']
            # predicted boxes are currently in [x1, y1, x2, y2] format
            pred_boxes = output['boxes']
            # convert both sets of boxes to [l, t, w, h] format
            pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
            true_boxes = u.convert_boxes_to_xywh(true_boxes)
            # normalize box pixel values between [0, 1]
            pred_boxes = u.norm_box_scale(pred_boxes)
            true_boxes = u.norm_box_scale(true_boxes)

            # concatenate confidences on the end of predicted boxes
            scores = output['scores']
            pred_boxes = u.cat_scores(pred_boxes, scores)
            # match every single prediction to a ground-truth (if any) and compute custom loss
            loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
            loss_tensor = c.compute_weighted_loss(loss_matches, loss_weights, iou_type)
            # get scalar tensor representing weighted sum of loss components
            # train_image_loss = loss_tensor[7]
            train_image_loss = loss_tensor[3]

            # backpropagate custom loss 
            train_image_loss.backward()
            # update accumulating epoch loss and number of total predictions seen
            train_epoch_loss += train_image_loss.item()
            num_preds += len(pred_boxes)

        # losses.backward()
        # data_iter.set_postfix(loss=losses)
        optimizer.step()
        data_iter.set_postfix(loss=train_epoch_loss / (i + 1))
    
    scheduler.step()
    train_epoch_loss /= num_preds
    return train_epoch_loss



def dummy_val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type):
    model.eval()
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

    with torch.no_grad():
        data_iter = tqdm(val_loader)
        for i, (images, targets) in enumerate(data_iter):
            if (i == 2):
                print(f"{i} batches validated successfully")
                break
            os.popen('rm -rf loss_matches_on_img')
            os.popen('rm -rf matches_on_img')

            cv2_images = images
            images = [(u.process_image(img)).to(device) for img in images]

            for t in targets:
                num_detections = t['num_detections']
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)
                    if k == 'boxes':
                        # slices through empty true boxes
                        t[k] = t[k][:num_detections, :]

            _, outputs = model(images)

            for j, output in enumerate(outputs):
                flight_id = str(targets[j]['flight_id'])
                frame_id = int(targets[j]['frame_id'])
                true_boxes = targets[j]['boxes']

                # access predicted boxes, convert them to [x, y, w, h] and concatenate scores
                pred_boxes = output['boxes']
                pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
                scores = output['scores']
                pred_boxes = u.cat_scores(pred_boxes, scores)
                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)

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
                # print(f"loss: {loss_tensor[7].item()}")
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