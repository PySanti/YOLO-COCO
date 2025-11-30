from pycocotools.coco import COCO
from utils.YOLODataset import YOLODataset
from utils.load_images_paths import load_images_paths
from utils.MACROS import TRAIN_ANN_FILE, VAL_ANN_FILE
from utils.render_yolo_image import render_yolo_image



if __name__ == "__main__":

    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    X_train_paths = load_images_paths("./dataset/train2017/train2017/")

    Y_val_wrapper = COCO(VAL_ANN_FILE)
    X_val_paths = load_images_paths("./dataset/val2017/val2017/")



    train_dataset = YOLODataset(X_train_paths, Y_train_wrapper)
    print(f"Cantidad de elementos de train: {len(X_train_paths)}")
    print(f"Cantidad de elementos de val: {len(X_val_paths)}")


    # muestra
#    p = train_dataset.__getitem__(6574)
#    print(p)
#    render_yolo_image(*p)
