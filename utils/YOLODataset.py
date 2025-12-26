from torch.utils.data import Dataset
from utils.encode_yolo_targets import encode_yolo_target
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
        image_tensor = self.transformer(image)
        image.close()
        image_annotation = get_image_target(get_image_id(self.X[idx]), self.Y)
        return image_tensor, encode_yolo_target(image_annotation, IMG_SIZE[0], IMG_SIZE[1], GRID_SIZE, NUM_CLASSES)

    def __len__(self):
        return len(self.X)
