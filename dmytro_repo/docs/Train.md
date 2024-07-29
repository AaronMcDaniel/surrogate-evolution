Update config.py to specify the dataset directory.
If the filesystem is read only, point to the writable location and create symlinks part1, part2 and part3 to the read only dataset.
It's recommended to use the SSD drive for the dataset storage.
Due to the dataset size, I conveted images to jpeg files to fit 2TB drive for each part, but training should work with the original files as well.
If the datset is conveted to jpegs, IMG_FORMAT in the config.py should be changed from 'png' to 'jpg'.


1) Training the transformation estimation model:
A few hours of training, up to 255th epoch:

python train_transformation.py train experiments/030_tr_tsn_rn34_w3_crop_borders.yaml

2) Predict transformation for the dataset frames:

Each step is quite slow, around 10-12 hours so recommended to run in parallel on separate GPUs.

python train_transformation.py predict_dataset_offsets experiments/030_tr_tsn_rn34_w3_crop_borders.yaml --part part1
python train_transformation.py predict_dataset_offsets experiments/030_tr_tsn_rn34_w3_crop_borders.yaml --part part2
python train_transformation.py predict_dataset_offsets experiments/030_tr_tsn_rn34_w3_crop_borders.yaml --part part3


3) Train the detection/tracking models:

This is the slowest part, may take up to a week for hrnet48 and a few days for other models.
It's recommended to run each training on separate GPU in parallel.

python train.py train experiments/120_gernet_m_b2_all.yaml
python train.py train experiments/120_hrnet32_all.yaml
python train.py train experiments/120_dla60_256_sgd_all_rerun.yaml
python train.py train experiments/130_hrnet48_all.yaml


Notes: hrnet48 used 18GB of VRAM during training and has been trained on 3090.
Should work fine with smaller amount of VRAM with the training batch size reduced from 16 to 8.
All models except of hrnet48 have been trained on the full dataset.


4) Export trained models (keep only weights, remove other data saved to checkpoints):

python train_transformation.py export_model experiments/030_tr_tsn_rn34_w3_crop_borders.yaml --epoch 255

python train.py export_model experiments/120_gernet_m_b2_all.yaml --epoch 2220
python train.py export_model experiments/120_hrnet32_all.yaml --epoch 2220
python train.py export_model experiments/120_dla60_256_sgd_all_rerun.yaml --epoch 2220
python train.py export_model experiments/130_hrnet48_all.yaml --epoch 2220


The model weights are exported to ../output/models/ and can be copied to ../data/models/ for inference.

