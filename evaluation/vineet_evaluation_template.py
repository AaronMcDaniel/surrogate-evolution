import os
import cv2
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import sys
from torch.cuda.amp import GradScaler, autocast
import torchvision
from sample_model import get_model_dict
from aot_dataset import AOTDataset, my_collate

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def prepare_data(batch_size=32):
    train_dataset = AOTDataset('train', string=1)
    val_dataset = AOTDataset('val', string=1)

    # Use multiple workers for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=4, pin_memory=True)
    return train_loader, val_loader

train_loader, val_loader = prepare_data(batch_size=10)

model_dict = get_model_dict("FCOS", 7, 7)
model = model_dict['model'].to(device)

batch_size = 10

# Define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

scaler = GradScaler()


def process_targets(targets):
    processed_targets = []
    for target in targets:
        num_detections = target['num_detections']
        boxes = target['boxes'][:num_detections]

        # Convert boxes from (x1, y1, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x2 = x1 + w
        y2 = y1 + h

        converted_boxes = torch.stack((x1, y1, x2, y2), dim=1)

        labels = target['labels'][:num_detections]
        
        processed_target = {
            'boxes': converted_boxes,
            'labels': labels
        }

        # Keep additional metadata if needed
        processed_target.update({
            'frame_id': target['frame_id'],
            'flight_id': target['flight_id'],
            'timestamp': target['timestamp'],
            'horizons': target['horizons'][:num_detections],
            'ranges': target['ranges'][:num_detections],
            'all_labels': target['all_labels'][:num_detections],
            'num_detections': num_detections,
            'path': target['path'],
            'image_id': target['image_id']
        })

        processed_targets.append(processed_target)

    return processed_targets

def move_targets_to_device(targets, device):
    device_targets = []
    for target in targets:
        device_target = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
        device_targets.append(device_target)
    return device_targets

def process_images(images):
    # Convert list of images to a single tensor
    images = torch.stack(images)
    
    # Permute to (B, C, H, W) and convert to float
    images = images.permute(0, 3, 1, 2).float()
    
    # Normalize the images to range [0, 1]
    images = images / 255.0
    
    return images

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_batch=None, print_freq=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = process_images(images).to(device)
        targets = process_targets(targets)
        targets = move_targets_to_device(targets, device)

        with autocast():
            loss_dict, detections = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache() 

        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

        if max_batch is not None and i == max_batch:
            break

    return running_loss / len(data_loader)

def draw_boxes(image, boxes, labels, color, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def evaluate_and_draw(model, data_loader, device):
    model.eval()
    output_folder = "test/outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = process_images(images).to(device)
            targets = process_targets(targets)
            targets = move_targets_to_device(targets, device)

            with autocast():
                detections = model(images, targets)

            for j, image in enumerate(images):
                image_np = image.permute(1, 2, 0).cpu().numpy()*255  # Convert to numpy array
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                # Draw target boxes in green
                draw_boxes(image_np, targets[j]['boxes'].cpu().numpy(), 'GT', (0, 255, 0))

                # Draw predicted boxes in red
                draw_boxes(image_np, detections[j]['boxes'].cpu().numpy(), 'Pred', (0, 0, 255))

                # Save the image
                output_path = os.path.join(output_folder, f'image_{i}_{j}.png')
                cv2.imwrite(output_path, image_np)

num_epochs = 1

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, max_batch=1000)
    evaluate_and_draw(model, val_loader, device)
    scheduler.step()

    print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss:.4f}")

print("Training complete.")