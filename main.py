from pycocotools.coco import COCO
import torch
import time
from utils.yolov1.YOLOv1 import YOLOv1
from utils.YOLODataset import YOLODataset
from torch import optim
from utils.yolov1.yolov1_loss import yolov1_loss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.MACROS import *
from utils.utils import  get_annotations_dict, load_images_paths

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])



    print("Cargando targets")

    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    # Las clases base de coco van del 1 .. 90 con saltos, clases que nunca aparecen
    # Se hizo un diccionario limpio de las clases (COCO_CLASSES_ES), sin embargo, las clases vienen en las annotations con el formato anterior ... Se deben transformar al activar la neurona en el target
    OLD_IDS = sorted(Y_train_wrapper.getCatIds()) # una lista de los ids antiguos
    NEW_IDS = {cid: i for i, cid in enumerate(OLD_IDS)} # obtienes un diccionario de id_viejo : id_nuevo
    X_train_paths = load_images_paths("./dataset/train2017/")
    train_dataset = YOLODataset(X_train_paths, get_annotations_dict(Y_train_wrapper), transformer, NEW_IDS)


    Y_val_wrapper = COCO(VAL_ANN_FILE)
    X_val_paths = load_images_paths("./dataset/val2017/")
    val_dataset = YOLODataset(X_val_paths, get_annotations_dict(Y_val_wrapper), transformer, NEW_IDS)

    TRAIN_LOADER = DataLoader(
            dataset=train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True
    )

    VAL_LOADER = DataLoader(
            dataset = val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=5,
            pin_memory=True,
            persistent_workers=True
            )

    model = YOLOv1().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0005)
    criterion = yolov1_loss


    print("Empezando entrenamiento")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        print("-------------")

        for i, (images, targets, ignored_boxes) in enumerate(TRAIN_LOADER):
            print("\r", end="")
            t1 = time.time()
            images = images.to(DEVICE)  # Move to GPU if available
            targets = targets.to(DEVICE)

            # Forward pass
            predictions = model(images)

            # Loss calculation
            loss = criterion(predictions, targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            print(f"({i+1}/{len(TRAIN_LOADER)}) - {(time.time()-t1)*(len(TRAIN_LOADER)-i-1):.1f} --- {sum(ignored_boxes)/BATCH_SIZE:.2f} IB", end="")
        print("\n")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (images, targets, ignored_boxes) in enumerate(VAL_LOADER):
                print("\r", end="")
                t1 = time.time()
                images, targets = images.to(DEVICE), targets.to(DEVICE)

                predictions = model(images)

                loss = criterion(predictions, targets)

                val_loss+=loss.item()

                print(f"({i+1}/{len(VAL_LOADER)}) - {(time.time()-t1)*(len(VAL_LOADER)-i-1):.1f}", end="")

        print("\n")
        print(f"Epoch {epoch+1}/{num_epochs} -  train_loss : {train_loss/len(TRAIN_LOADER):.4f} - val_loss : {val_loss / len(VAL_LOADER)}")

