import os
import subprocess
import random
import numpy as np
import pickle
import aot_dataset as ad


# Setup vars
train_target_images = 40000
val_target_images = 10000
train_part = 'part1'
val_part = 'part2'
aws_base_command = "s3://airborne-obj-detection-challenge-training/"
train_output_dir = "aot_data/train/Images"
val_output_dir = "aot_data/val/Images"
train_label_dir = 'aot_data/train/Labels'
val_label_dir = 'aot_data/val/Labels'
train_pickle_path = '/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/train/Labels/part1_STRING_TENSORV2_labels.pkl'
val_pickle_path = '/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/val/Labels/part2_STRING_TENSORV2_labels.pkl'

# Make sure out dirs exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Random rng seed
rng = np.random.default_rng(seed=42)

# Used to count the num images in flight folder
def get_image_count(flight_dir):
    command = f"aws s3 ls {aws_base_command}{flight_dir} --no-sign-request --recursive | wc -l"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())

# Used to verify the train/val splits are the same for group members
def count_images_in_dir(directory):
    total = 0
    
    for flight_dir in os.listdir(directory):
        flight_path = os.path.join(directory, flight_dir)
        
        if os.path.isdir(flight_path):
            for root, dirs, files in os.walk(flight_path):
                image_files = [file for file in files if file.endswith(('.png'))]
                total += len(image_files)
    return total

# Used to download the selected flight
def download_flight(flight_id, part, output_dir):
    flight_dir = os.path.join(output_dir, "Images", flight_id)

    # If already downloaded, skip it
    if os.path.exists(flight_dir):
        print(f"Flight {flight_id} already exists, skipping...")
        return 0
    command = f"aws s3 sync {aws_base_command}{part}/Images/{flight_id} {output_dir}/Images/{part}{flight_id} --no-sign-request"
    subprocess.run(command, shell=True)

    # Get num images for the flight
    num_images = len([f for f in os.listdir(flight_dir) if os.path.isfile(os.path.join(flight_dir, f))])
    return num_images

# Used to randomly select flights and download until the target num images is reached
def download_random_flights(part, target_images, output_dir):
    command = f"aws s3 ls {aws_base_command}{part}/Images/ --no-sign-request"
    flights = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.splitlines()
    flight_list = [flight.split()[-1] for flight in flights]
    
    if not flight_list:
        raise ValueError(f"No flights available in {part}. Please check the S3 path or permissions.")
    
    total_images = 0
    if os.path.exists(output_dir):

        # Count num images already downloaded if resuming from download
        total_images = sum([len(files) for r, d, files in os.walk(output_dir)])
    
    selected_flights = []

    while total_images < target_images:
        flight_id = rng.choice(flight_list)

        # Ensure flight is not selected again
        flight_list.remove(flight_id)
        
        # Download flight and get num images in flight
        num_images = download_flight(flight_id, part, output_dir)
        
        # num_images is only 0 if the flight has already been downloaded
        # If it wasnt already downloaded, add it to the count
        if num_images > 0:
            total_images += num_images
            selected_flights.append(flight_id)
        
        print(f"Downloaded {total_images} images so far from {len(selected_flights)} flights.")
    
    print(f"Completed downloading {total_images} images from {len(selected_flights)} flights in {part}.")

# def rename_folders(base_dir):
#     for flight in os.listdir(base_dir):
#         flight_folder = os.path.join(base_dir, flight)
#         if os.path.isdir(flight_folder):
#             new_name = os.path.join(base_dir, f'{part}{flight}')
#             os.rename(flight_folder, new_name)

def rename_folders(base_dir):
    for flight in os.listdir(base_dir):
        flight_folder = os.path.join(base_dir, flight)
        if os.path.isdir(flight_folder):
            new_name = os.path.join(base_dir, flight[2:])  # Remove the first 8 characters
            os.rename(flight_folder, new_name)

def shrink_pickle(label_dir, pickle_path, part):
    with open(pickle_path, 'rb') as f:
            labels = pickle.load(f)
    new_labels = []
    for l in labels:
        l = ad.eval_label(l)
        path = l['path']
        path = os.path.join((label_dir.split('/')[0]) + '/' + (label_dir.split('/')[1]) + '/' + path)
        if os.path.exists(path):
            serialized_label = repr(l)
            new_labels.append(serialized_label)
    os.makedirs(label_dir, exist_ok=True)
    output_path = os.path.join(label_dir, f'{part}_STRING_TENSORV2_SHRUNKEN_labels.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(new_labels, f)
    
    print(f"Saved {len(new_labels)} labels to {output_path}")

# download_random_flights(train_part, train_target_images, train_output_dir)
# download_random_flights(val_part, val_target_images, val_output_dir)

# Num train images should be 40474
print(count_images_in_dir('/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/train/Images'))
# print(count_images_in_dir('/storage/ice-shared/vip-vvk/data/AOT'))

# Num val images should be 10789
print(count_images_in_dir('/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/val/Images'))
# print(count_images_in_dir('/storage/ice-shared/vip-vvk/data/AOT'))

# TEAM NEEDS TO RENAME FOLDERS TOO
# rename_folders(train_output_dir)
# rename_folders(val_output_dir)

# SHRINKING PICKLE FILES
shrink_pickle(train_label_dir, train_pickle_path, train_part)
shrink_pickle(val_label_dir, val_pickle_path, val_part)

import pickle

with open('/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/train/Labels/part1_STRING_TENSORV2_SHRUNKEN_labels.pkl', 'rb') as file:
    data = pickle.load(file)

# Print or inspect the data
print(data)


with open('/home/hice1/hweston3/scratch/surrogate-evolution/aot_data/val/Labels/part2_STRING_TENSORV2_SHRUNKEN_labels.pkl', 'rb') as file:
    data = pickle.load(file)

# Print or inspect the data
print(data)