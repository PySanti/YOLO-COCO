from pycocotools.coco import COCO
from utils.YOLODataset import YOLODataset
from utils.load_images_paths import load_images_paths
from utils.MACROS import TRAIN_ANN_FILE



if __name__ == "__main__":

    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    X_train_paths = load_images_paths("./dataset/train2017/train2017/")

    train_dataset = YOLODataset(X_train_paths, Y_train_wrapper)

    print(train_dataset.__getitem__(5))
