# Airborne Object Tracking (AOT) dataset

The Airborne Object Tracking (AOT) dataset is a collection of flight sequences collected onboard aerial vehicles with high-resolution cameras. To generate those sequences, two aircraft are equipped with sensors and fly _planned_ encounters.

The trajectories are designed to create a wide distribution of distances, closing velocities, and approach angles. 

In addition to the so-called planned aircraft, AOT also contains other unplanned airborne objects, which may be present in the sequences.
Those objects are also labeled but their distance information is not available.

Airborne objects usually appear quite small at the distances which are relevant for early detection: 0.01% of the image size on average, down to a few pixels in area (compared to common object detection datasets, which exhibit objects covering more considerable portion of the image). This makes AOT a new and challenging dataset for the detection and tracking of potential aerial collision threats. 


------

## Accessing Training Dataset

The complete training dataset size is ~>11TB.
You can also download partial dataset (500G) using `partial=True` flag in `Dataset`. It includes all the frames with valid encounter of planned airborne object.

You can access the dataset in public S3 bucket hosted at `s3://airborne-obj-detection-challenge-training/`.

**In order to ease access to you as participant, we have added some helper scripts which will help you download the dataset on need basis.
You can simply load all the ground_truth files and download dataset flight by flight.**

Extensive example for that are covered in getting started notebook which can be accessed in this [Repository](/-/blob/master/data/dataset-playground.ipynb) as well as on [Colab](https://colab.research.google.com/drive/1B5Gevpg6GIlfMRRfiG79V8Foz13_ncUr).

### 💻 **[Run on Colab](https://colab.research.google.com/drive/1B5Gevpg6GIlfMRRfiG79V8Foz13_ncUr) | 💪 [Run on Local](/-/blob/master/data/dataset-playground.ipynb)**

------

## Dataset directory structure

You can download and use the files based on your preference. The dataset exploration and default structure followed by this starter kit is as follow:

```bash
data
├── part1
│   ├── ImageSets
│   │   └── groundtruth.json
│   └── Images
│       ├── 1497343b9d90411db5c305e785be9032
│       │   ├── 15580184151527797371497343b9d90411db5c305e785be9032.png
│       │   ├── 15580184153446871561497343b9d90411db5c305e785be9032.png
│       │   ├── 15580184154352609941497343b9d90411db5c305e785be9032.png
│       │   └── 15580184155474411311497343b9d90411db5c305e785be9032.png
│       │   └── [...]
│       ├── 1f9a42f2d2194622b845bf5ad9ba1fce
│       │   ├── 15445285838485478651f9a42f2d2194622b845bf5ad9ba1fce.png
│       │   ├── 15445285839226252081f9a42f2d2194622b845bf5ad9ba1fce.png
│       │   ├── 15445285840353354091f9a42f2d2194622b845bf5ad9ba1fce.png
│       │   ├── 15445285841304961601f9a42f2d2194622b845bf5ad9ba1fce.png
│       │   ├── [...]
│  
├── part2
│   └── ImageSets
│       └── groundtruth.json
│   └── Images
│       ├── [...]
│  
├── part3
│   ├── ImageSets
│   │   └── groundtruth.json
│   └── Images
│       └── f3b3af98f63543a0965fab8b005b13c7
│           ├── 1568218496792027402f3b3af98f63543a0965fab8b005b13c7.png
│           ├── 1568218496891165702f3b3af98f63543a0965fab8b005b13c7.png
│           ├── 1568218496986100575f3b3af98f63543a0965fab8b005b13c7.png
│           ├── 1568218497082896562f3b3af98f63543a0965fab8b005b13c7.png
```

-------

## Ground Truth

Ground truth (present in `ImageSets` folder) contains all the relevant information regarding airborne objects, their locations, bbox and so on.
While the `Images` folder have accompanied images for your training code to work on.

Before we start, let's check the vocabulary we will need to understand the dataset:

* `flights` (a.k.a. `samples` in ground truth):<br>
  One flight is typically 2 minutes video at 10 fps i.e. 1200 images. Each of the frames are present in `Images/{{flight_id}}/` folder. These files are typically 3-4mb each.


* `frame` (a.k.a. `entity` in ground truth):<br>
  This is the most granular unit on which dataset can be sampled. Each frame have information timestamp, frame_id, and label `is_above_horizon`.
  There can be multiple entries for same frame in `entity` when multiple Airborne objects are present.<br>
  When an Airborne object following information is available as well:
  - `id` -> signifies unique ID of this object (for whole frame)
  - `bbox` -> it contains 4 floats signifying `[left, top, width, height]`
  - `blob['range_distance_m']` -> distance of airborne object
  - `labels['is_above_horizon']` -> details below
  - (derived) `planned` -> for the planned objects `range_distance_m` is available
    

* `is_above_horizon`:<br>
  It is marked as 1 when Airborne object is above horizon and -1 when it is below horizon. When unclear, it is marked as 0.

Example for frame level data (multiple per frame):

```json
{
    'time': 1550844897919368155,
    'blob': {
        'frame': 480,
        'range_distance_m': nan # signifies, it was an unplanned object
    },
    'id': 'Bird2',
    'bb': [1013.4, 515.8, 6.0, 6.0],
    'labels': {'is_above_horizon': 1},
    'flight_id': '280dc81adbb3420cab502fb88d6abf84',
    'img_name': '1550844897919368155280dc81adbb3420cab502fb88d6abf84.png'
}
```

## Downloading the dataset

### Using helper data access library

This library will provide you quick access and download only the files you require.

#### Initializing

```python
from core.dataset import Dataset
dataset = Dataset(local_path='data/part1', s3_path='s3://airborne-obj-detection-challenge-training/part1/', prefix='part1')
dataset.add(local_path='data/part2', s3_path='s3://airborne-obj-detection-challenge-training/part2/', prefix='part2')
dataset.add(local_path='data/part3', s3_path='s3://airborne-obj-detection-challenge-training/part3/', prefix='part3')
```

Example for partial dataset:
```python
from core.dataset import Dataset
dataset = Dataset(local_path='data/part1', s3_path='s3://airborne-obj-detection-challenge-training/part1/', prefix='part1', partial=True)
dataset.add(local_path='data/part2', s3_path='s3://airborne-obj-detection-challenge-training/part2/', prefix='part2', partial=True)
dataset.add(local_path='data/part3', s3_path='s3://airborne-obj-detection-challenge-training/part3/', prefix='part3', partial=True)
```

NOTE: You don't need to have `groundtruth.json` pre-downloaded, it will automatically download, save and load them for you.

#### Playing with Flights, Frames and more

```python
flight_ids = dataset.get_flight_ids()
flight = dataset.get_flight(flight_ids[0]) # Loading single flight

airborne_objects = flight.get_airborne_objects() # Get all airborne objects
airborne_objects.location # Location of object in whole flight

frames = flight.frames # All the frames of the flight
airborne_objects_in_frame = frames.detected_objects
airborne_objects_location_in_frame = frames.detected_object_locations

# Check out the dataset exploration notebook for more...
[...]
```

#### Download complete dataset for individual flight 
```python
flight.download()
```

#### Extras

_Reading images and generating videos_

```python
frame.image()
frame.image(type='cv2')
frame.image(type='pil')
frame.image(type='numpy')

[...]

video_path = frame.generate_video(speed_x=6) # for 6x speed video
```

You can contribute to helper library as well by raising PR to this repository.

### Using `aws s3`

In case you want to download the whole dataset, you can do it using following commands (in `data/` directory).

```bash
aws s3 sync s3://airborne-obj-detection-challenge-training/part1 part1/ --no-sign-request
aws s3 sync s3://airborne-obj-detection-challenge-training/part1 part2/ --no-sign-request
aws s3 sync s3://airborne-obj-detection-challenge-training/part1 part3/ --no-sign-request
```

## Additional details about the dataset

Please note that the entire training contains ~ 12TB. To experiment with less flights / images, please review `valid_encounters_maxRange700_maxGap3_minEncLen30.json` (provided in each training folder), which contains information about encounters (defined in the Benchmarks section). For each encounter, we provide the corresponding sequence (sub-folder) name, relevant image names and additional information on distance statistics of aircraft in the encounter, if the encounter is below or above horizon and its length in frames. The provided baseline used only the images (frames) that correspond to the encounters, those are a good start for participant’s first training.

-----

👋 In case you have any doubts or need help, you can reach out to us via Challenge [Discussions](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge/discussion) or [Discord](https://discord.gg/BT9uegr).
