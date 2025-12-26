from pycocotools.coco import COCO
import torch
import time
from utils.YOLOv1 import YOLOv1
from utils.YOLODataset import YOLODataset
from torch import optim
from utils.yolov1_loss import yolov1_loss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.MACROS import *
from utils.utils import load_images_paths

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    print("Cargando targets")
    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    X_train_paths = load_images_paths("./dataset/train2017/")
    train_dataset = YOLODataset(X_train_paths, Y_train_wrapper, transformer)


    #Y_val_wrapper = COCO(VAL_ANN_FILE)
    #X_val_paths = load_images_paths("./dataset/val2017/val2017/")

    TRAIN_LOADER = DataLoader(
            dataset=train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
    )

    model = YOLOv1().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = yolov1_loss  # Your loss function from earlier

    print("Empezando entrenamiento")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (images, targets) in enumerate(TRAIN_LOADER):
            t1 = time.time()
            images = images.to(DEVICE)  # Move to GPU if available
            targets = targets.to(DEVICE)

            # Forward pass
            predictions = model(images)

            # Loss calculation
            loss = criterion(predictions, targets, num_classes=90)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"({i}/{len(train_dataset)//BATCH_SIZE}) - tiempo de procesamiento de batch : {time.time()-t1}", end="\r")

        print("\n\n")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/(len(train_dataset)//BATCH_SIZE):.4f}")
