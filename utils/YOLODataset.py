from torch.utils.data import Dataset
from utils.encode_yolo_target import encode_yolov1
from utils.utils import get_image_id
from PIL import Image
from utils.utils import get_image_target
from utils.MACROS import *

class YOLODataset(Dataset):
    def __init__(self, X, Y, transformer) -> None:
        super().__init__()
        self.X = X # paths list
        self.Y = Y # target wrapper
        self.transformer = transformer

    def __getitem__(self, idx) :
        """
            Recordar que las imagenes tienen unos ids (que se encuentra en su nombre)
            Mientras que los targets tienen otro id
        """
        image = Image.open(self.X[idx]).convert('RGB') 
        prev_size = image.size
        image_tensor = self.transformer(image)
        image.close()
        image_annotation = get_image_target(get_image_id(self.X[idx]), self.Y)
        encoded_target, ignored_boxes = encode_yolov1(prev_size, image_annotation, IMG_SIZE, GRID_SIZE, NUM_CLASSES)
        return image_tensor, encoded_target, ignored_boxes

    def __len__(self):
        return len(self.X)
