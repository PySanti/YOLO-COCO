import enum
from matplotlib.pyplot import plot
from pycocotools.coco import COCO
from utils.utils import load_images_paths
from utils.MACROS import COCO_CLASSES_ES, TRAIN_ANN_FILE, VAL_ANN_FILE
from utils.utils import get_dataset_classes_count
from utils.utils import plot_class_distribution
from utils.utils import render_yolo_image
from utils.YOLODataset import YOLODataset



if __name__ == "__main__":

    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    X_train_paths = load_images_paths("./dataset/train2017/train2017/")
    train_dataset = YOLODataset(X_train_paths, Y_train_wrapper)

    Y_val_wrapper = COCO(VAL_ANN_FILE)
    X_val_paths = load_images_paths("./dataset/val2017/val2017/")

    
    train_classes_count = get_dataset_classes_count(X_train_paths, Y_train_wrapper)
    non_app = [x for x,y in enumerate(train_classes_count) if x!=0 and y == 0]
    print(f"Las clases que no aparecen en train son : {non_app}")
    plot_class_distribution(train_classes_count,COCO_CLASSES_ES, "Distribucion de train")


    val_classes_count = get_dataset_classes_count(X_val_paths, Y_val_wrapper)
    non_app = [x for x,y in enumerate(val_classes_count) if x!=0 and y == 0]
    print(f"Las clases que no aparecen en val son : {non_app}")
    plot_class_distribution(val_classes_count, COCO_CLASSES_ES, "Distribucion de val")
