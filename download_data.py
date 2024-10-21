import os
import pickle
import subprocess
import random
import numpy as np
import aot_dataset as ad
import glob
train_target_images = 40000
val_target_images = 10000
train_part = 'part1'
val_part = 'part2'
aws_base_command = "s3://airborne-obj-detection-challenge-training/"
train_output_dir = "/storage/ice-shared/vip-vvk/data/AOT/aot_data/train/Images"
val_output_dir = "/storage/ice-shared/vip-vvk/data/AOT/aot_data/val/Images"
train_label_dir = '/storage/ice-shared/vip-vvk/data/AOT/aot_data/train/Labels'
val_label_dir = '/storage/ice-shared/vip-vvk/data/AOT/aot_data/val/Labels'
train_pickle_path = '/storage/ice-shared/vip-vvk/data/AOT/aot_data/train/Labels/part1_STRING_TENSORV2_labels.pkl'
val_pickle_path = '/storage/ice-shared/vip-vvk/data/AOT/aot_data/val/Labels/part2_STRING_TENSORV2_labels.pkl'


def remove_unused_images(output_dir,pickle_path):
    print("Processing Labels...")
    with open(pickle_path,"rb") as f:
        labels = pickle.load(f)


    included_paths = list(map(lambda x: ad.eval_label(x)["path"],labels))
    print("Done processing labels...")
    #Loop through paths and delete those that aren't present in the pickle file
    count = 0
    output_dir = "/".join(output_dir.split("/")[:-1])
    for file in glob.glob("Images/**/*.png",root_dir=output_dir):
        if file not in included_paths:
            count+=1
            os.remove(os.path.join(output_dir,file))

    print(count)

def shrink_pickle(label_dir, pickle_path, part,output_dir):
    #This shrink pickle won't work
    print("Processing Labels...")
    with open(pickle_path,"rb") as f:
        labels = pickle.load(f)
    
    labels = list(map(ad.eval_label,labels))
    print("Done processing labels")
    output_dir = "/".join(output_dir.split("/")[:-1])
    chosen_labels = []
    for label in labels:
        #Only add to chosen labels if the path exists.
        path = label["path"]
        total_path = os.path.join(output_dir,path)
        if os.path.exists(total_path):
            chosen_labels.append(label)
    print("Num labels:",len(chosen_labels))
    print("Making new labels...")
    os.makedirs(label_dir, exist_ok=True)
    output_path = os.path.join(label_dir, f'{part}_STRING_TENSORV2_SHRUNKEN_labels.pkl')
    with open(output_path, 'wb') as f:
        chosen_labels = list(map(repr,chosen_labels))
        pickle.dump(chosen_labels, f)

def download_data(output_dir,label_dir,pickled_labels,num_images,part):
    rng = np.random.default_rng(seed=42)
    print("Processing Labels...")
    with open(pickled_labels,"rb") as f:
        labels = pickle.load(f)
    
    labels = list(map(ad.eval_label,labels))
    print("Done processing labels")
    chosen_labels = []
    downloaded_images = {}
    for i, label_object in enumerate(labels):
        flight_id = label_object["flight_id"]

        if flight_id not in downloaded_images:
            downloaded_images[flight_id] = set()
        downloaded_images[flight_id].add(i) #Add index in labels instead
        
    print("Now downloading images....")
    flight_ids = list(downloaded_images.keys())
    current_images = 0
    while current_images < num_images:
        flight_id = rng.choice(flight_ids)
        label_indexes =  downloaded_images[flight_id]
        label_objects = [labels[i] for i in label_indexes]
        images = set(map(lambda x: x["path"].split("/")[-1],label_objects))
        num_taken = len(images)
        out_dir_with_flight = os.path.join(output_dir,flight_id)
        if os.path.exists(out_dir_with_flight):
            continue #No need to redownload existing images, just assume it has already worked
        os.makedirs(out_dir_with_flight)
        #Contains part
        command = f"aws s3 sync {aws_base_command}{part}/Images/{flight_id[5:]} {out_dir_with_flight} --no-sign-request"
        subprocess.run(command, shell=True)

        print("Removing unused images...")
        for file in os.listdir(out_dir_with_flight):
            file_path = os.path.join(out_dir_with_flight,file)
            if file not in images and os.path.exists(file_path):
                os.remove(file_path)
        chosen_labels.extend(list(label_objects))
        current_images += num_taken
        print(f"Currently have {current_images}....")




if __name__ == "__main__":
    print("Starting")
    # download_data(val_output_dir,val_label_dir,val_pickle_path,val_target_images,val_part)
    # download_data(train_output_dir,train_label_dir,train_pickle_path,train_target_images,train_part)
    # shrink_pickle(val_label_dir,val_pickle_path,val_part,val_output_dir)
    # shrink_pickle(train_label_dir,train_pickle_path,train_part,train_output_dir)
    # remove_unused_images(train_output_dir,train_pickle_path)



#What I need to do is download the entire flight directory

#Then go through and remove unused images

#So I need to have a data structure, like a dict, containing a flight and a set of image keys
#Then I need to download the flight one by one and delete unincluded images
#I can do so
#Then I need to go through and delete images in the flights that are not 