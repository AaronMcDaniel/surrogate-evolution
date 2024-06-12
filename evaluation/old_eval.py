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
