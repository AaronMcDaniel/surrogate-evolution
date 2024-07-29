DATA_DIR = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data'
SRC_DATA_DIR = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data'
TEST_DATA_DIR = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/Images/' #part10a4f022fdf2e4bcf9efa189e617c4b3a'
IMG_FORMAT = 'png'
DETECTOR_ONLY = False

UPPER_BOUND_MIN_DIST = 330
UPPER_BOUND_MAX_DIST = 700
UPPER_BOUND_MAX_DIST_SELECTED_TRAIN = 1000
MAX_PREDICT_DISTANCE = 2000

OFFSET_SCALE = 256.0

MIN_OBJECT_AREA = 100
IS_MATCH_MIN_IOU_THRESH = 0.2
IS_NO_MATCH_MAX_IOU_THRESH = 0.02
MIN_SECS = 3.0

CLASSES = [
    'None',
    'Airborne',
    'Airplane',
    'Bird',
    'Drone',
    'Flock',
    'Helicopter'
]

NB_CLASSES = len(CLASSES)


TRANSFORM_MODEL = '030_tr_tsn_rn34_w3_crop_borders'
TRANSFORM_MODEL_EPOCH = 255
TRANSFORM_MODEL_FOLD = 0
