# OLD IMPLEMENTATION OF EVALUATE:
# def evaluate(genome, gen, ind_num, num_epochs, data_dir, out_dir):
#     # ethan handles data
#     train = AOTDetection(root='REPLACE', annFile='REPLACE', transform=ToTensor())
#     test = AOTDetection(root='REPLACE', annFile='REPLACE', transform=ToTensor())
#     batch_size = 50
#     train_loader = DataLoader(train, batch_size, shuffle=True, pin_memory=True)
#     test_loader = DataLoader(test, batch_size, shuffle=False)
#     val_loader = DataLoader(test, batch_size, shuffle=False, pin_memory=True)
#     model = Codec.decode(genome) # this will be implemented 

#     device = torch.device('cuda') if torch.cuda.is_avauilable() else torch.device('cpu')
#     model.to(device) # moves model to gpu
#     params = [p for p in model.parameters() if p.requires_grad] # collects trainable params
#     # loss function, optimizer, and other parameters will be defined by the decoded genome
#     optimizer = optim.SGD(params, lr=0.001, momentum = 0.9, weight_decay = 0.0005)
#     criterion = nn.CrossEntropyLoss()
#     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#     accuracies = []
#     losses = []
#     val_accuracies = []
#     val_losses = []
#     # train
#     for epoch in range(num_epochs):
#         model.train()
#         for images, targets in train_loader:
#             # moves images and targets to gpu
#             images = list(image.to(device) for image in images)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#             outputs = model(images) # returns losses and detections
#             loss = criterion(outputs, targets)
#             # backprop and update params
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             _, predicted = torch.max(outputs.data, 1)
#         acc = (predicted == torch.tensor([t['labels'] for t in targets]).to(device)).sum().item() / len(targets)
#         accuracies.append(acc)
#         losses.append(loss.item())
#         lr_scheduler.step()
#         # eval model on validation 
#         val_loss = 0.0
#         val_acc = 0.0
#         with torch.no_grad():
#             for images, targets, in val_loader:
#                 images = list(image.to(device) for image in images)
#                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#                 outputs = model(images)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 val_acc += (predicted == torch.tensor([t['labels'] for t in targets]).to(device)).sum().item() / len(targets)
#             total = targets.size(0)
#             correct = (predicted == targets).sum().item()
#             val_acc += correct / total
#             val_accuracies.append(val_acc / len(val_loader))
#             val_losses.append(val_loss / len(val_loader))
#         print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format(epoch+1, num_epochs, loss.item(), val_loss, acc ,val_acc))

#     # eval model on test
#     model.eval() # sets model to evaluation mode (disables dropout, batch normalization, etc.)
#     with torch.no_grad(): # just inferencing, so no gradient
#         correct = 0
#         total = 0
#         y_true = []
#         y_pred = []
#         for images, targets in test_loader:
#             images = list(image.to(device) for image in images)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#             ouputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += len(targets)
#             correct += (predicted == torch.tensor([t['labels'] for t in targets]).to(device)).sum().item()
#             y_true.extend([t['labels'].item() for t in targets])
#             y_pred.extend(predicted.cpu().numpy())
#     print(classification_report(y_true, y_pred))


# OLD IMPLEMENTATION OF MATCH_BOXES: Greedy algorithm, matches pairs based on current best IoU, without regard to solution that could come later
# ensures that truth-prediction matches are 1-to-1

    # # iterate through predicted boxes
    # for pi in range(pred_boxes.shape[0]):

    #     # don't consider predictions that don't meet the confidence threshold
    #     if (pred_boxes[pi][4] < conf_thresh):
    #         continue

    #     # keep track of the best matching true bbox and iou
    #     best_iou = 0
    #     best_ti = -1
        
    #     # iterate through true boxes
    #     for ti in range(true_boxes.shape[0]):

    #         # do not consider true boxes that have already been matched
    #         if ti not in matched_truths_indices:

    #             # retrieve iou from matrix
    #             iou = iou_matrix[pi, ti].item()

    #             # check that the iou meets the threshold and is better than the best_iou
    #             if iou >= iou_thresh and iou > best_iou:
    #                 best_ti = ti
    #                 best_iou = iou
        
    #     # if a true bbox match was found, add the matches, and record the indices
    #     if best_ti != -1:
    #         matched_preds.append(pred_boxes[pi])
    #         matched_truths.append(true_boxes[best_ti])
    #         matched_preds_indices.add(pi)
    #         matched_truths_indices.add(best_ti)

    # for pi in range(pred_boxes.shape[0]):
    #     if pi not in matched_preds_indices:
    #         fp.append(pred_boxes[pi])
    
    # for ti in range(true_boxes.shape[0]):
    #     if ti not in matched_truths_indices:
    #         fn.append(true_boxes[ti])

    # return matched_preds, matched_truths, fp, fn

    # # # matches is a tensor containing pairs of indices where the IoU between predicted and true boxes exceeds the threshold
    # # matches = (iou_matrix >= iou_thresh).nonzero(as_tuple=False)

    # # # nume1() returns the number of elements in the tensor to check if there are any matches
    # # if matches.nume1() > 0:

# OLD IMPLEMENTATION OF MATCH_BOXES: does not filter duplicates properly

    #     # extracts the unique indices of matched predicted boxes and makes a list
    #     pred_matched = matches[:, 0].unique().tolist()
    #     # extracts the unique indices of matched true boxes and makes a list
    #     true_matched = matches[:, 1].unique().tolist()

    #     for match in matches:
    #         # unpacks indices of predicted and true match
    #         pred_i, true_i = match

    #         # gets associated bboxes
    #         matched_preds.append(pred_boxes[pred_i])
    #         matched_truths.append(true_boxes[true_i])
        
    #     # iterates through the predicted boxes
    #     for i in range(pred_boxes.shape[0]):
    #         # if the predicted box does not have corresponding true match above the threshold, it is a false positive
    #         if i not in pred_matched:
    #             fp.append(pred_boxes[i])
        
    #     # iterates through the true boxes
    #     for i in range(true_boxes.shape[0]):
    #         # if the true box does not have corresponding predicted match above the threshold, it is a false negative
    #         if i not in true_matched:
    #             fn.append(true_boxes[i])
    # else:
    #     # in the case no matches are found, all predictions are false positives, and all truths are false negatives
    #     fp = pred_boxes
    #     fn = true_boxes
    
    # return matched_preds, matched_truths, fp, fn