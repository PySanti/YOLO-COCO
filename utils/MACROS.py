TRAIN_ANN_FILE = "./dataset/annotations_trainval2017/annotations/instances_train2017.json"
VAL_ANN_FILE = "./dataset/annotations_trainval2017/annotations/instances_val2017.json"
IMG_SIZE = (224,224)
GRID_SIZE = 7
BATCH_SIZE = 128
NUM_CLASSES = 80
ANNOTATIONS_REQUIRED = [
        "bbox",
        "category_id",
        "image_id"
        ]