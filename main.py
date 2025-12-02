import enum
from pycocotools.coco import COCO
import torch
from utils.utils import load_images_paths, plot_3d_tensor
from utils.MACROS import TRAIN_ANN_FILE, VAL_ANN_FILE
from utils.YOLO import YOLO
from utils.YOLODataset import YOLODataset
from torch import optim
from utils.yolo_loss import yolo_loss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    Y_train_wrapper = COCO(TRAIN_ANN_FILE)
    X_train_paths = load_images_paths("./dataset/train2017/")
    train_dataset = YOLODataset(X_train_paths, Y_train_wrapper, transformer)


    #Y_val_wrapper = COCO(VAL_ANN_FILE)
    #X_val_paths = load_images_paths("./dataset/val2017/val2017/")

    TRAIN_LOADER = DataLoader(
            dataset=train_dataset, 
            batch_size=64,
            shuffle=True,
        #            num_workers=8,
        #    pin_memory=True,
        #   persistent_workers=True
    )

    model = YOLO(grid_size=7, num_classes=90, num_anchors=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = yolo_loss  # Your loss function from earlier

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in TRAIN_LOADER:
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
